from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import config as project_config
from src.data import (
    ensure_val_split,
    instances_for_image,
    load_coco,
    resolve_image,
    three_channel,
)


def _augment(image: np.ndarray, masks: list[np.ndarray]):
    if random.random() < 0.5:
        image = image[:, ::-1].copy()
        masks = [m[:, ::-1].copy() for m in masks]

    if random.random() < 0.5:
        image = image[::-1, :].copy()
        masks = [m[::-1, :].copy() for m in masks]

    return image, masks


class FilamentMaskRCNNDataset(Dataset):
    def __init__(self, train: bool = True, augment: bool = False):
        self.coco = load_coco()
        self.augment = augment

        val_ids = set(ensure_val_split(self.coco))
        self.rows = []

        for info in self.coco.imgs.values():
            stem = Path(info["file_name"]).stem
            is_val = stem in val_ids

            if train and is_val:
                continue
            if (not train) and (not is_val):
                continue

            path = resolve_image(project_config.train_images, info["file_name"])
            if path is not None:
                self.rows.append((info, path))

        self.rows.sort(key=lambda row: Path(row[0]["file_name"]).stem)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        info, path = self.rows[idx]

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read image: {path}")

        image = three_channel(gray)
        masks = instances_for_image(self.coco, info["id"])

        if self.augment:
            image, masks = _augment(image, masks)

        h, w = image.shape[:2]
        clean_masks = []
        boxes = []

        for mask in masks:
            mask = (mask > 0).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue

            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(min(w, int(xs.max()) + 1))
            y2 = float(min(h, int(ys.max()) + 1))

            if x2 <= x1 or y2 <= y1:
                continue

            clean_masks.append(mask)
            boxes.append([x1, y1, x2, y2])

        if clean_masks:
            masks_tensor = torch.from_numpy(
                np.stack(clean_masks).astype(np.uint8)
            )
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.ones((len(clean_masks),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
        }

        # IMPORTANT: do not resize here. Torchvision's Mask R-CNN transform
        # resizes the image AND boxes/masks together to config.image_size.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float() / 255.0

        return image_tensor, target
