from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import dagshub
import matplotlib.pyplot as plt
import mlflow
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
from src.metrics import aggregate, panoptic_quality

CKPT = "facebook/mask2former-swin-tiny-coco-instance"
DATA_SEED = 42
CROP_ATTEMPTS = 8
DATASET_MIN_AREA = 32
BEST_WORST_TOP_K = 10


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


def _augment(
    image: np.ndarray, masks: list[np.ndarray], rng: np.random.RandomState
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Применяет случайные геометрические аугментации к изображению и маскам синхронно.
    params:
        image: изображение (H, W, 3)
        masks: список бинарных масок (H, W)
        rng: генератор случайных чисел
    returns:
        кортеж из преобразованных image и masks
    """
    if rng.rand() < 0.5:
        image = image[:, ::-1]
        masks = [m[:, ::-1] for m in masks]
    k = rng.randint(0, 4)
    if k:
        image = np.rot90(image, k)
        masks = [np.rot90(m, k) for m in masks]
    return image, masks


class Magfilo(Dataset):
    """Датасет MAGFiLO из train-изображений."""

    def __init__(
        self, size: int = 512, min_area: int = DATASET_MIN_AREA, augment: bool = False
    ) -> None:
        """Составляет список train-изображений, исключая валидационный сплит.
        params:
            size: сторона случайного кропа
            min_area: минимальная площадь маски, чтобы она сохранилась в кропе
            augment: использовать случайные геометрические аугментации (flip/повороты)
        """
        self.size = size
        self.min_area = min_area
        self.augment = augment
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
        """Возвращает одно изображение и его маски.
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
                    image, masks = crop, cropped
                    break
            else:
                image = image[y : y + self.size, x : x + self.size]
                masks = cropped
        else:
            masks = [m for m in masks if int(m.sum()) >= self.min_area]
        if self.augment:
            image, masks = _augment(image, masks, self.rng)
        return {"image": image, "masks": masks}


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
    """Делает инференс модели на одном изображении.
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


def init_dagshub() -> bool:
    """Инициализирует трекинг MLflow на DagsHub.
    returns:
        True, если трекинг настроен, иначе False
    """
    dc = config.dagshub
    if not dc.configured:
        print("DagsHub logging disabled: set DAGSHUB_TOKEN in .env")
        return False
    try:
        dagshub.init(repo_owner=dc.repo_owner, repo_name=dc.repo_name, mlflow=True)
        mlflow.set_experiment(dc.experiment_name)
        return True
    except Exception as exc:
        print(f"WARNING: DagsHub init failed, continuing without logging: {exc}")
        return False


def _val_items() -> list[tuple]:
    """Формирует список валидационных изображений вместе с COCO-аннотациями и путями к файлам.
    returns:
        список кортежей (coco, info, path) валидационных изображений
    """
    coco = load_coco()
    val = ensure_val_split(coco)
    items = []
    for info in coco.imgs.values():
        if image_stem(info["file_name"]) not in val:
            continue
        path = resolve_image(config.train_images, info["file_name"])
        if path is not None:
            items.append((coco, info, path))
    return items


@torch.no_grad()
def _eval_val(model, processor, device, eval_samples: int = 0) -> float:
    """Считает Home PQ модели на валидационных изображениях.
    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        eval_samples: число случайных изображений для оценки (0 — все)
    returns:
        агрегированный PQ по валидации (0, если нет изображений)
    """
    items = _val_items()
    if eval_samples and len(items) > eval_samples:
        items = random.Random(DATA_SEED).sample(items, eval_samples)
    rows = []
    for coco, info, path in tqdm(items, desc="eval val PQ"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gt = instances_for_image(coco, info["id"])
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(gt, pred))
    if not rows:
        return 0.0
    return float(aggregate(rows)["pq"])


@torch.no_grad()
def _eval_train(model, processor, device, eval_samples: int = 0) -> float:
    """Считает Home PQ модели на тренировочных изображениях.
    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        eval_samples: число случайных изображений для оценки (0 — все)
    returns:
        агрегированный PQ по трейну (0, если нет изображений)
    """
    coco = load_coco()
    val = ensure_val_split(coco)
    items = []
    for info in coco.imgs.values():
        if image_stem(info["file_name"]) in val:
            continue
        path = resolve_image(config.train_images, info["file_name"])
        if path is not None:
            items.append((coco, info, path))
    if eval_samples and len(items) > eval_samples:
        items = random.Random(DATA_SEED).sample(items, eval_samples)
    rows = []
    for coco, info, path in tqdm(items, desc="eval train PQ"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gt = instances_for_image(coco, info["id"])
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(gt, pred))
    if not rows:
        return 0.0
    return float(aggregate(rows)["pq"])


def _save_best_worst(model, processor, device, run_dir: Path) -> None:
    """Прогоняет модель по всей валидации и сохраняет top-k худших/лучших снимков с overlay GT/pred.
    params:
        model: модель с весами best.pt
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        run_dir: папка эксперимента (outputs/m2f/experiments/<run_name>)
    """
    results = []
    for coco, info, path in tqdm(_val_items(), desc="best/worst val"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_masks(model, processor, img, device)
        pq = panoptic_quality(instances_for_image(coco, info["id"]), pred)["pq"]
        results.append((pq, info, path))
    results.sort(key=lambda r: r[0])
    for name, seq in (
        ("bad", results[:BEST_WORST_TOP_K]),
        ("good", results[-BEST_WORST_TOP_K:]),
    ):
        folder = run_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for pq, info, path in seq:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for m in instances_for_image(coco, info["id"]):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
            for m in predict_masks(model, processor, gray, device):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 255, 0), 2)
            cv2.imwrite(str(folder / f"pq{pq:.3f}_{path.stem}.png"), vis)


def log_train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "m2f",
    augment: bool = False,
) -> dict:
    """Обучает Mask2Former на MAGFiLO и логирует эксперимент в DagsHub.
    params:
        size: сторона кропа при обучении
        epochs: число эпох
        max_steps: ограничение по числу шагов (0 — без ограничения)
        accum: число шагов накопления градиента
        lr: скорость обучения
        workers: число воркеров DataLoader
        eval_val_samples: число случайных изображений для val PQ (0 — вся валидация)
        eval_train_samples: число случайных изображений для train PQ (0 — не считать)
        run_name: имя MLflow run на DagsHub
        augment: использовать аугментации (flip/повороты) при обучении
    returns:
        словарь с историей train_loss, train_pq и val_pq по эпохам
    """
    if not config.train_annotations.exists():
        raise SystemExit(
            f"No data at {config.train_annotations}. On Kaggle: Add Data → filament-segmentation-2026"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build(size)
    model.to(device).train()
    data = Magfilo(size=size, augment=augment)
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
    run_dir = out / "experiments" / run_name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    best_pq = -1.0
    history: dict = {"train_loss": [], "val_pq": [], "train_pq": []}
    run = None
    if init_dagshub():
        try:
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
                    "checkpoint": CKPT,
                    "size": size,
                    "epochs": epochs,
                    "accum": accum,
                    "lr": lr,
                    "weight_decay": 0.05,
                    "workers": workers,
                    "augment": augment,
                }
            )
        except Exception as exc:
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None
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
        avg_loss = running / max(seen, 1)
        history["train_loss"].append(avg_loss)
        model.eval()
        val_pq = _eval_val(model, processor, device, eval_val_samples)
        if eval_train_samples:
            train_pq = _eval_train(model, processor, device, eval_train_samples)
        else:
            train_pq = None
        model.train()
        history["val_pq"].append(val_pq)
        history["train_pq"].append(train_pq)
        if run is not None:
            metrics = {"train_loss": avg_loss, "val_pq": val_pq}
            if train_pq is not None:
                metrics["train_pq"] = train_pq
            mlflow.log_metrics(metrics, step=epoch)
        epoch_ckpt = weights_dir / f"epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(), "size": size, "epoch": epoch}, epoch_ckpt)
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_pq": train_pq,
                "val_pq": val_pq,
            }
        )
        if val_pq > best_pq:
            best_pq = val_pq
            torch.save(
                {"model": model.state_dict(), "size": size, "epoch": epoch},
                run_dir / "best.pt",
            )
        print(f"epoch {epoch} loss={avg_loss:.4f} -> {ckpt}")
        if max_steps and step_i >= max_steps:
            break
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_pq", "val_pq"])
        w.writeheader()
        w.writerows(metrics_rows)
    processor.save_pretrained(out / "processor")
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
        _save_best_worst(model, processor, device, run_dir)
    if run is not None:
        _log_training_artifacts(model, ckpt, history)
        mlflow.end_run()
    return history


def _log_training_artifacts(model, ckpt: Path, history: dict) -> None:
    """Сохраняет в активный MLflow-ран графики обучения, модель и чекпоинт.
    params:
        model: обученная модель Mask2Former
        ckpt: путь к чекпоинту last.pt
        history: словарь с историей train_loss, val_pq и train_pq по эпохам
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="train_loss", marker="o")
        valid_val = [
            (e, v) for e, v in zip(epochs, history["val_pq"]) if v is not None
        ]
        valid_train = [
            (e, v) for e, v in zip(epochs, history["train_pq"]) if v is not None
        ]
        if valid_val or valid_train:
            ax2 = ax.twinx()
            if valid_val:
                ax2.plot(*zip(*valid_val), label="val_pq", marker="s", color="tab:orange")
            if valid_train:
                ax2.plot(*zip(*valid_train), label="train_pq", marker="^", color="tab:green")
            ax2.set_ylabel("pq")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train_loss")
        fig.legend(loc="upper right")
        fig.tight_layout()
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)
        mlflow.pytorch.log_model(model, "mask2former_model", serialization_format="pt2")
    except Exception as exc:
        print(f"WARNING: MLflow artifacts failed, ckpt still logged: {exc}")
    mlflow.log_artifact(str(ckpt))


def train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "m2f",
    augment: bool = False,
) -> Path:
    """Обучает Mask2Former на MAGFiLO: выполняет обучение через log_train (с DagsHub-логированием, если оно настроено), сохраняет чекпоинт и процессор.
    params:
        size: сторона кропа при обучении
        epochs: число эпох
        max_steps: ограничение по числу шагов (0 — без ограничения)
        accum: число шагов накопления градиента
        lr: скорость обучения
        workers: число воркеров DataLoader
        eval_val_samples: число случайных изображений для val PQ (0 — вся валидация)
        eval_train_samples: число случайных изображений для train PQ (0 — не считать)
        run_name: имя MLflow run на DagsHub
        augment: использовать аугментации (flip/повороты) при обучении
    returns:
        путь к сохранённому чекпоинту last.pt
    """
    log_train(
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        accum=accum,
        lr=lr,
        workers=workers,
        eval_val_samples=eval_val_samples,
        eval_train_samples=eval_train_samples,
        run_name=run_name,
        augment=augment,
    )
    return config.models_path / "last.pt"
