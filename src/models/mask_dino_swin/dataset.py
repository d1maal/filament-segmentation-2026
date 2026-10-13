"""Батчинг для Swin: ресайз, ImageNet-нормализация и маски инстансов."""
from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Нормализует трёхканальное uint8-изображение статистиками ImageNet.
    params:
        image: массив (H, W, 3) uint8
    returns:
        массив (H, W, 3) float32
    """
    arr = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    return (arr - mean) / std


def collate(size: int, batch: list[dict]) -> dict:
    """Собирает батч из элементов Magfilo: тензор пикселей и маски GT.
    params:
        size: сторона квадратных входов
        batch: список элементов датасета с image и masks
    returns:
        словарь с pixel_values (B, 3, S, S) и mask_labels (список тензоров)
    """
    images = []
    for item in batch:
        resized = cv2.resize(item["image"], (size, size), interpolation=cv2.INTER_LINEAR)
        images.append(torch.from_numpy(normalize_image(resized).transpose(2, 0, 1)))
    mask_labels = []
    for item in batch:
        kept = []
        for mask in item["masks"]:
            resized = cv2.resize(
                (mask > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
            ).astype(np.float32)
            if resized.any():
                kept.append(torch.from_numpy(resized))
        mask_labels.append(torch.stack(kept) if kept else torch.zeros(0, size, size))
    return {"pixel_values": torch.stack(images), "mask_labels": mask_labels}
