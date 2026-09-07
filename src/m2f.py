from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from src.data import (
    OUTPUTS,
    TRAIN_ANN,
    TRAIN_IMAGES,
    ensure_val_split,
    image_stem,
    instances_for_image,
    load_coco,
    resolve_image,
    three_channel,
)

CKPT = "facebook/mask2former-swin-tiny-coco-instance"


def build(size: int = 512):
    processor = Mask2FormerImageProcessor.from_pretrained(CKPT, size={"height": size, "width": size})
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CKPT,
        id2label={0: "filament"},
        label2id={"filament": 0},
        ignore_mismatched_sizes=True,
    )
    return processor, model


class Magfilo(Dataset):
    def __init__(self, size: int, min_area: int = 32) -> None:
        self.size = size
        self.min_area = min_area
        self.rng = np.random.RandomState(42)
        self.coco = load_coco()
        val = ensure_val_split(self.coco)
        self.rows = []
        for info in self.coco.imgs.values():
            if image_stem(info["file_name"]) in val:
                continue
            path = resolve_image(TRAIN_IMAGES, info["file_name"])
            if path is not None:
                self.rows.append((info, path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        info, path = self.rows[idx]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image, masks = three_channel(gray), instances_for_image(self.coco, info["id"])
        h, w = image.shape[:2]
        if h > self.size and w > self.size:
            for _ in range(8):
                y = int(self.rng.randint(0, h - self.size + 1))
                x = int(self.rng.randint(0, w - self.size + 1))
                crop = image[y : y + self.size, x : x + self.size]
                cropped = [m[y : y + self.size, x : x + self.size] for m in masks]
                cropped = [m for m in cropped if int(m.sum()) >= self.min_area]
                if cropped:
                    return {"image": crop, "masks": cropped}
            image = image[y : y + self.size, x : x + self.size]
            masks = cropped
        return {"image": image, "masks": [m for m in masks if int(m.sum()) >= self.min_area]}


def collate(processor, batch: list[dict]) -> dict:
    encoded = processor(images=[b["image"] for b in batch], return_tensors="pt")
    _, _, h, w = encoded["pixel_values"].shape
    mask_labels, class_labels = [], []
    for item in batch:
        kept = []
        for mask in item["masks"]:
            if mask.shape != (h, w):
                mask = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
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


def _masks_from_result(result: dict, score_thr: float, min_area: int) -> list[np.ndarray]:
    seg = result.get("segmentation")
    infos = result.get("segments_info") or []
    if seg is None:
        return []

    masks = []
    if torch.is_tensor(seg) and seg.ndim == 3:
        for i, info in enumerate(infos):
            if float(info.get("score", 0)) < score_thr:
                continue
            mask = (seg[i] > 0).cpu().numpy().astype(np.uint8)
            if int(mask.sum()) >= min_area:
                masks.append(mask)
        return masks

    id_map = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
    for info in infos:
        if float(info.get("score", 0)) < score_thr:
            continue
        mask = (id_map == int(info["id"])).astype(np.uint8)
        if int(mask.sum()) >= min_area:
            masks.append(mask)
    return masks


def _postprocess(processor, outputs, size: tuple[int, int], score_thr: float, min_area: int) -> list[np.ndarray]:
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_thr,
        target_sizes=[size],
        return_binary_maps=True,
    )[0]
    return _masks_from_result(result, score_thr, min_area)


@torch.no_grad()
def predict_masks(model, processor, image: np.ndarray, device, score_thr: float = 0.5, min_area: int = 80) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inputs = {k: v.to(device) for k, v in processor(images=three_channel(gray), return_tensors="pt").items()}
    outputs = model(**inputs)
    return _postprocess(processor, outputs, (h, w), score_thr, min_area)


@torch.no_grad()
def predict_masks_grid(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thrs: list[float],
    min_areas: list[int],
) -> dict[tuple[float, int], list[np.ndarray]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inputs = {k: v.to(device) for k, v in processor(images=three_channel(gray), return_tensors="pt").items()}
    outputs = model(**inputs)

    predictions = {}
    for score_thr in score_thrs:
        masks = _postprocess(processor, outputs, (h, w), float(score_thr), 0)
        areas = [int(mask.sum()) for mask in masks]
        for min_area in min_areas:
            predictions[(float(score_thr), int(min_area))] = [
                mask for mask, area in zip(masks, areas) if area >= int(min_area)
            ]
    return predictions


def train(size: int = 512, epochs: int = 8, max_steps: int = 0, accum: int = 8, lr: float = 1e-4, workers: int = 0) -> Path:
    if not TRAIN_ANN.exists():
        raise SystemExit(f"No data at {TRAIN_ANN}. On Kaggle: Add Data → filament-segmentation-2026")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build(size)
    model.to(device).train()
    data = Magfilo(size=size)
    if not data:
        raise SystemExit("Empty train split")
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=workers, collate_fn=lambda b: collate(processor, b))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        autocast = lambda: torch.amp.autocast("cuda", enabled=device.type == "cuda")
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    out = OUTPUTS / "m2f"
    out.mkdir(parents=True, exist_ok=True)
    step_i = 0
    for epoch in range(1, epochs + 1):
        running, seen, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}"), start=1):
            with autocast():
                loss = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[m.to(device) for m in batch["mask_labels"]],
                    class_labels=[c.to(device) for c in batch["class_labels"]],
                ).loss / accum
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
