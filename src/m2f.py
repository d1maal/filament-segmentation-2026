from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

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

CHECKPOINTS = {
    "tiny": "facebook/mask2former-swin-tiny-coco-instance",
    "small": "facebook/mask2former-swin-small-coco-instance",
}


def build(
    size: int = 512,
    backbone: str = "tiny",
    no_object_weight: float = 0.1,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
):
    if backbone not in CHECKPOINTS:
        raise ValueError(f"Unknown backbone: {backbone}. Use one of: {', '.join(CHECKPOINTS)}")

    checkpoint = CHECKPOINTS[backbone]
    processor = Mask2FormerImageProcessor.from_pretrained(
        checkpoint,
        size={"height": size, "width": size},
    )
    config = Mask2FormerConfig.from_pretrained(
        checkpoint,
        id2label={0: "filament"},
        label2id={"filament": 0},
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return processor, model


class Magfilo(Dataset):
    def __init__(
        self,
        size: int,
        min_area: int = 32,
        object_crop_prob: float = 0.7,
        negative_crop_prob: float = 0.1,
        augment: bool = True,
    ) -> None:
        if object_crop_prob < 0 or negative_crop_prob < 0 or object_crop_prob + negative_crop_prob > 1:
            raise ValueError("Crop probabilities must be >= 0 and sum to <= 1")

        self.size = size
        self.min_area = min_area
        self.object_crop_prob = object_crop_prob
        self.negative_crop_prob = negative_crop_prob
        self.augment = augment
        self.full_image = False
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

    def set_full_image(self, enabled: bool) -> None:
        self.full_image = enabled

    def _random_xy(self, h: int, w: int) -> tuple[int, int]:
        y = int(self.rng.randint(0, h - self.size + 1))
        x = int(self.rng.randint(0, w - self.size + 1))
        return y, x

    def _crop(self, image: np.ndarray, masks: list[np.ndarray], y: int, x: int) -> tuple[np.ndarray, list[np.ndarray]]:
        crop = image[y : y + self.size, x : x + self.size]
        cropped = [m[y : y + self.size, x : x + self.size] for m in masks]
        cropped = [m for m in cropped if int(m.sum()) >= self.min_area]
        return crop, cropped

    def _object_crop(self, image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        h, w = image.shape[:2]
        valid = [m for m in masks if int(m.sum()) >= self.min_area]
        if not valid:
            y, x = self._random_xy(h, w)
            return self._crop(image, masks, y, x)

        target = valid[int(self.rng.randint(0, len(valid)))]
        ys, xs = np.where(target > 0)
        center_y = (int(ys.min()) + int(ys.max())) // 2
        center_x = (int(xs.min()) + int(xs.max())) // 2

        jitter = max(1, self.size // 8)
        center_y += int(self.rng.randint(-jitter, jitter + 1))
        center_x += int(self.rng.randint(-jitter, jitter + 1))

        y = int(np.clip(center_y - self.size // 2, 0, h - self.size))
        x = int(np.clip(center_x - self.size // 2, 0, w - self.size))
        return self._crop(image, masks, y, x)

    def _negative_crop(self, image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        h, w = image.shape[:2]
        fallback = None
        for _ in range(16):
            y, x = self._random_xy(h, w)
            crop, cropped = self._crop(image, masks, y, x)
            fallback = (crop, cropped)
            if not cropped:
                return crop, []
        return fallback

    def _augment(self, image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        if not self.augment:
            return image, masks

        if self.rng.rand() < 0.5:
            image = np.flip(image, axis=1)
            masks = [np.flip(m, axis=1) for m in masks]
        if self.rng.rand() < 0.5:
            image = np.flip(image, axis=0)
            masks = [np.flip(m, axis=0) for m in masks]

        k = int(self.rng.randint(0, 4))
        if k:
            image = np.rot90(image, k)
            masks = [np.rot90(m, k) for m in masks]

        return np.ascontiguousarray(image), [np.ascontiguousarray(m) for m in masks]

    def __getitem__(self, idx: int) -> dict:
        info, path = self.rows[idx]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image, masks = three_channel(gray), instances_for_image(self.coco, info["id"])
        h, w = image.shape[:2]

        if not self.full_image and h > self.size and w > self.size:
            sample = self.rng.rand()
            if sample < self.object_crop_prob:
                image, masks = self._object_crop(image, masks)
            elif sample < self.object_crop_prob + self.negative_crop_prob:
                image, masks = self._negative_crop(image, masks)
            else:
                y, x = self._random_xy(h, w)
                image, masks = self._crop(image, masks, y, x)
        else:
            masks = [m for m in masks if int(m.sum()) >= self.min_area]

        image, masks = self._augment(image, masks)
        return {"image": image, "masks": masks}


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


def _instances_from_result(result: dict, score_thr: float, min_area: int) -> list[tuple[np.ndarray, float]]:
    seg = result.get("segmentation")
    infos = result.get("segments_info") or []
    if seg is None:
        return []

    instances = []
    if torch.is_tensor(seg) and seg.ndim == 3:
        for i, info in enumerate(infos):
            score = float(info.get("score", 0))
            if score < score_thr:
                continue
            mask = (seg[i] > 0).cpu().numpy().astype(np.uint8)
            if int(mask.sum()) >= min_area:
                instances.append((mask, score))
        return instances

    id_map = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
    for info in infos:
        score = float(info.get("score", 0))
        if score < score_thr:
            continue
        mask = (id_map == int(info["id"])).astype(np.uint8)
        if int(mask.sum()) >= min_area:
            instances.append((mask, score))
    return instances


def _postprocess_instances(
    processor,
    outputs,
    size: tuple[int, int],
    score_thr: float,
    min_area: int,
) -> list[tuple[np.ndarray, float]]:
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_thr,
        target_sizes=[size],
        return_binary_maps=True,
    )[0]
    return _instances_from_result(result, score_thr, min_area)


def _postprocess(processor, outputs, size: tuple[int, int], score_thr: float, min_area: int) -> list[np.ndarray]:
    return [mask for mask, _ in _postprocess_instances(processor, outputs, size, score_thr, min_area)]


@torch.no_grad()
def _predict_scored(model, processor, image: np.ndarray, device, score_thr: float, min_area: int = 0):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inputs = {k: v.to(device) for k, v in processor(images=three_channel(gray), return_tensors="pt").items()}
    outputs = model(**inputs)
    return _postprocess_instances(processor, outputs, (h, w), score_thr, min_area)


@torch.no_grad()
def predict_masks(model, processor, image: np.ndarray, device, score_thr: float = 0.5, min_area: int = 80) -> list[np.ndarray]:
    return [mask for mask, _ in _predict_scored(model, processor, image, device, score_thr, min_area)]


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    end = length - tile_size
    if starts[-1] != end:
        starts.append(end)
    return starts


def _mask_similarity(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if not inter:
        return 0.0
    area_a = a.sum(dtype=np.int64)
    area_b = b.sum(dtype=np.int64)
    union = area_a + area_b - inter
    iou = float(inter / union) if union else 0.0
    containment = float(inter / min(area_a, area_b)) if min(area_a, area_b) else 0.0
    return max(iou, containment)


def _merge_candidates(
    candidates: list[tuple[np.ndarray, float]],
    overlap_thr: float,
    min_area: int,
) -> list[np.ndarray]:
    merged: list[tuple[np.ndarray, float]] = []
    for mask, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        matched = False
        for i, (kept, kept_score) in enumerate(merged):
            if _mask_similarity(mask, kept) >= overlap_thr:
                merged[i] = (np.logical_or(mask, kept).astype(np.uint8), max(score, kept_score))
                matched = True
                break
        if not matched:
            merged.append((mask, score))

    return [mask for mask, _ in merged if int(mask.sum()) >= min_area]


@torch.no_grad()
def predict_masks_tiled(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float = 0.3,
    min_area: int = 80,
    tile_size: int = 1024,
    overlap: int = 256,
    merge_overlap: float = 0.6,
    include_full: bool = True,
) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    candidates: list[tuple[np.ndarray, float]] = []

    if include_full:
        candidates.extend(_predict_scored(model, processor, gray, device, score_thr, 0))

    ys = _tile_starts(h, tile_size, overlap)
    xs = _tile_starts(w, tile_size, overlap)
    margin = overlap // 2

    for y in ys:
        for x in xs:
            tile = gray[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            tile_h, tile_w = tile.shape[:2]
            instances = _predict_scored(model, processor, tile, device, score_thr, 0)

            for mask, score in instances:
                ys_mask, xs_mask = np.where(mask > 0)
                if not len(xs_mask):
                    continue

                center_y = float(ys_mask.mean())
                center_x = float(xs_mask.mean())
                top_margin = 0 if y == 0 else margin
                left_margin = 0 if x == 0 else margin
                bottom_margin = 0 if y + tile_h >= h else margin
                right_margin = 0 if x + tile_w >= w else margin

                if not (
                    top_margin <= center_y < tile_h - bottom_margin
                    and left_margin <= center_x < tile_w - right_margin
                ):
                    continue

                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y : y + tile_h, x : x + tile_w] = mask
                candidates.append((full_mask, score))

    return _merge_candidates(candidates, merge_overlap, min_area)


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


def _make_scheduler(optimizer, updates_per_epoch: int, epochs: int, warmup_epochs: int):
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(0, updates_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1e-3, float(step + 1) / warmup_updates)
        if total_updates <= warmup_updates:
            return 1.0
        progress = min(1.0, (step - warmup_updates) / (total_updates - warmup_updates))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
    backbone: str = "tiny",
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    full_image_epochs: int = 3,
    no_object_weight: float = 0.08,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
    warmup_epochs: int = 2,
    backbone_lr_mult: float = 0.1,
) -> Path:
    if not TRAIN_ANN.exists():
        raise SystemExit(f"No data at {TRAIN_ANN}. On Kaggle: Add Data → filament-segmentation-2026")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build(
        size=size,
        backbone=backbone,
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
    )
    model.to(device).train()

    data = Magfilo(
        size=size,
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        augment=True,
    )
    if not data:
        raise SystemExit("Empty train split")

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        collate_fn=lambda b: collate(processor, b),
    )
    backbone_params = list(model.model.pixel_level_module.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=0.05,
    )
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, accum)))
    scheduler = _make_scheduler(opt, updates_per_epoch, epochs, warmup_epochs)

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
        use_full_image = full_image_epochs > 0 and epoch > epochs - full_image_epochs
        data.set_full_image(use_full_image)
        mode = "full" if use_full_image else "crop"

        running, seen, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch} [{mode}]"), start=1):
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
                scheduler.step()

            running += float(loss.detach().cpu()) * accum
            seen += 1
            step_i += 1
            if max_steps and step_i >= max_steps:
                break

        if seen and step % accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "size": size,
            "epoch": epoch,
            "backbone": backbone,
            "no_object_weight": no_object_weight,
            "dice_weight": dice_weight,
            "mask_weight": mask_weight,
            "object_crop_prob": object_crop_prob,
            "negative_crop_prob": negative_crop_prob,
            "full_image_epochs": full_image_epochs,
            "warmup_epochs": warmup_epochs,
            "backbone_lr_mult": backbone_lr_mult,
            "lr": lr,
        }
        last = out / "last.pt"
        epoch_path = out / f"epoch_{epoch:02d}.pt"
        torch.save(checkpoint, last)
        torch.save(checkpoint, epoch_path)

        backbone_lr = opt.param_groups[0]["lr"]
        head_lr = opt.param_groups[1]["lr"]
        print(
            f"epoch {epoch} mode={mode} loss={running / max(seen, 1):.4f} "
            f"lr={head_lr:.2e} backbone_lr={backbone_lr:.2e} -> {epoch_path}"
        )

        if max_steps and step_i >= max_steps:
            break

    processor.save_pretrained(out / "processor")
    return out / "last.pt"
