from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from src.config import config
from src.data import (
    IMAGE_SIZE,
    ensure_val_split,
    image_stem,
    instances_for_image,
    load_coco,
    resolve_image,
    three_channel,
)

CKPT = "facebook/mask2former-swin-tiny-coco-instance"
DATA_SEED = 42
CROP_ATTEMPTS = 8
DATASET_MIN_AREA = 32


def build(size: int = 512):
    """Собирает процессор и модель Mask2Former для сегментации филаментов.
    params:
        size: сторона входных квадратных изображений
    returns:
        кортеж из процессора и модели
    """
    processor = Mask2FormerImageProcessor.from_pretrained(
        CKPT, size={"height": size, "width": size}
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CKPT,
        id2label={0: "filament"},
        label2id={"filament": 0},
        ignore_mismatched_sizes=True,
    )
    return processor, model


class Magfilo(Dataset):
    """Датасет MAGFiLO из train-изображений."""

    def __init__(self, size: int = 512, min_area: int = DATASET_MIN_AREA) -> None:
        """Составляет список train-изображений, исключая валидационный сплит.
        params:
            size: сторона случайного кропа
            min_area: минимальная площадь маски, чтобы она сохранилась в кропе
        """
        self.size = size
        self.min_area = min_area
        self.rng = np.random.RandomState(DATA_SEED)
        self.coco = load_coco()
        val = ensure_val_split(self.coco)
        self.rows = []
        for info in self.coco.imgs.values():
            if image_stem(info["file_name"]) in val:
                continue
            path = resolve_image(config.train_images, info["file_name"])
            if path is not None:
                self.rows.append((info, path))

    def __len__(self) -> int:
        """Возвращает число изображений в датасете.
        returns:
            количество train-изображений
        """
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Возвращает одно изображение и его маски: читает файл, преобразует в три канала и режет случайный кроп с достаточным числом масок.
        params:
            idx: индекс элемента
        returns:
            словарь с трёхканальным изображением и списком бинарных масок
        """
        info, path = self.rows[idx]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        image, masks = three_channel(gray), instances_for_image(self.coco, info["id"])
        h, w = image.shape[:2]
        if h > self.size and w > self.size:
            for _ in range(CROP_ATTEMPTS):
                y = int(self.rng.randint(0, h - self.size + 1))
                x = int(self.rng.randint(0, w - self.size + 1))
                crop = image[y : y + self.size, x : x + self.size]
                cropped = [m[y : y + self.size, x : x + self.size] for m in masks]
                cropped = [m for m in cropped if int(m.sum()) >= self.min_area]
                if cropped:
                    return {"image": crop, "masks": cropped}
            image = image[y : y + self.size, x : x + self.size]
            masks = cropped
        return {
            "image": image,
            "masks": [m for m in masks if int(m.sum()) >= self.min_area],
        }


def collate(processor, batch: list[dict]) -> dict:
    """Кодирует батч изображений процессором и формирует целевые маски и классы для Mask2Former.
    params:
        processor: процессор Mask2Former
        batch: список элементов датасета
    returns:
        словарь с pixel_values, mask_labels и class_labels
    """
    encoded = processor(images=[b["image"] for b in batch], return_tensors="pt")
    _, _, h, w = encoded["pixel_values"].shape
    mask_labels, class_labels = [], []
    for item in batch:
        kept = []
        for mask in item["masks"]:
            if mask.shape != (h, w):
                mask = cv2.resize(
                    (mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            if mask.any():
                kept.append(torch.from_numpy(mask.astype(np.float32)))
        if kept:
            mask_labels.append(torch.stack(kept))
            class_labels.append(torch.zeros(len(kept), dtype=torch.long))
        else:
            mask_labels.append(torch.zeros(0, h, w))
            class_labels.append(torch.zeros(0, dtype=torch.long))
    encoded["mask_labels"] = mask_labels
    encoded["class_labels"] = class_labels
    return encoded


@torch.no_grad()
def predict_masks(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float = 0.5,
    min_area: int = 80,
) -> list[np.ndarray]:
    """Делает инференс модели на одном изображении: преобразует вход, прогоняет сеть и возвращает бинарные маски инстансов с фильтрами по порогу и площади.
    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale-изображение (или BGR)
        device: устройство выполнения вычислений
        score_thr: минимальный скор детекции
        min_area: минимальная площадь маски в пикселях
    returns:
        список бинарных масок инстансов
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inputs = {
        k: v.to(device)
        for k, v in processor(images=three_channel(gray), return_tensors="pt").items()
    }
    result = processor.post_process_instance_segmentation(
        model(**inputs),
        threshold=score_thr,
        target_sizes=[(h, w)],
        return_binary_maps=True,
    )[0]
    seg, infos = result.get("segmentation"), result.get("segments_info") or []
    if seg is None:
        return []
    masks = []
    if torch.is_tensor(seg) and seg.ndim == 3:
        for i, info in enumerate(infos):
            if float(info.get("score", 0)) < score_thr:
                continue
            binary = (seg[i] > 0).cpu().numpy().astype(np.uint8)
            if int(binary.sum()) >= min_area:
                masks.append(binary)
        return masks
    id_map = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
    for info in infos:
        if float(info.get("score", 0)) < score_thr:
            continue
        binary = (id_map == int(info["id"])).astype(np.uint8)
        if int(binary.sum()) >= min_area:
            masks.append(binary)
    return masks


def train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
) -> Path:
    """Обучает Mask2Former на MAGFiLO: создаёт датасет и загрузчик, прогоняет цикл с накоплением градиента и mixed-precision, сохраняет чекпоинт и процессор после каждой эпохи.
    params:
        size: сторона кропа при обучении
        epochs: число эпох
        max_steps: ограничение по числу шагов (0 — без ограничения)
        accum: число шагов накопления градиента
        lr: скорость обучения
        workers: число воркеров DataLoader
    returns:
        путь к сохранённому чекпоинту last.pt
    """
    if not config.train_annotations.exists():
        raise SystemExit(
            f"No data at {config.train_annotations}. On Kaggle: Add Data → filament-segmentation-2026"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build(size)
    model.to(device).train()
    data = Magfilo(size=size)
    if not data:
        raise SystemExit("Empty train split")
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        collate_fn=lambda b: collate(processor, b),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        autocast = lambda: torch.amp.autocast("cuda", enabled=device.type == "cuda")
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    out = config.models_path
    out.mkdir(parents=True, exist_ok=True)
    step_i = 0
    for epoch in range(1, epochs + 1):
        running, seen, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}"), start=1):
            with autocast():
                loss = (
                    model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[m.to(device) for m in batch["mask_labels"]],
                        class_labels=[c.to(device) for c in batch["class_labels"]],
                    ).loss
                    / accum
                )
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * accum
            seen += 1
            step_i += 1
            if max_steps and step_i >= max_steps:
                break
        if seen and step % accum:
            scaler.step(opt)
            scaler.update()
        ckpt = out / "last.pt"
        torch.save({"model": model.state_dict(), "size": size, "epoch": epoch}, ckpt)
        print(f"epoch {epoch} loss={running / max(seen, 1):.4f} -> {ckpt}")
        if max_steps and step_i >= max_steps:
            break
    processor.save_pretrained(out / "processor")
    return out / "last.pt"
