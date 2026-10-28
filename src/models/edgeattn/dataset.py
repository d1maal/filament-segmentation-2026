from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from src.config import config
from src.data import (
    ensure_val_split,
    image_stem,
    load_coco,
    resolve_image,
    three_channel,
)
from src.eval.val_items import _val_items
from src.models.m2f.dataset import DATA_SEED, DATASET_MIN_AREA, Magfilo

NORM_MEAN = 0.5
NORM_STD = 0.5


def normalize_hwc(image: np.ndarray) -> np.ndarray:
    """Нормирует изображение (H, W, C) из [0, 255] в [-1, 1].
    params:
        image: изображение uint8 (H, W, C)
    returns:
        float32-массив (H, W, C)
    """
    return (image.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD


def make_input(gray: np.ndarray, in_ch: int) -> np.ndarray:
    """Готовит вход модели из grayscale-изображения.
    params:
        gray: grayscale-изображение (H, W)
        in_ch: число каналов модели (1 — grayscale, 3 — raw/CLAHE/high-pass)
    returns:
        float32-массив (H, W, in_ch) в [-1, 1]
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected grayscale (H, W), got {gray.shape}")
    if in_ch == 1:
        image = gray[..., None]
    elif in_ch == 3:
        image = three_channel(gray)
    else:
        raise ValueError(f"in_ch must be 1 or 3, got {in_ch}")
    return normalize_hwc(image)


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Объединяет маски инстансов в одну бинарную карту.
    params:
        masks: список бинарных масок инстансов
        shape: целевое разрешение (H, W)
    returns:
        бинарная маска (H, W) типа uint8
    """
    target: np.ndarray = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.shape != shape:
            binary = cv2.resize(
                binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            )
        target |= binary
    return target


class SemanticMagfilo(Magfilo):
    """MAGFiLO для семантической сегментации: цель — union масок инстансов."""

    def __init__(
        self,
        in_ch: int = 3,
        size: int = 512,
        min_area: int = DATASET_MIN_AREA,
        object_crop_prob: float = 0.7,
        negative_crop_prob: float = 0.1,
        augment: bool = True,
    ) -> None:
        """Создаёт семантический датасет по train-изображениям MAGFiLO.
        params:
            in_ch: число входных каналов (1 — grayscale, 3 — raw/CLAHE/high-pass)
            size: сторона обучающего кропа
            min_area: минимальная площадь маски в кропе
            object_crop_prob: доля кропов вокруг филамента
            negative_crop_prob: доля пустых кропов
            augment: геометрические аугментации (flip/повороты)
        """
        if in_ch not in (1, 3):
            raise ValueError(f"in_ch must be 1 or 3, got {in_ch}")
        super().__init__(
            size=size,
            min_area=min_area,
            object_crop_prob=object_crop_prob,
            negative_crop_prob=negative_crop_prob,
            augment=augment,
        )
        self.in_ch = in_ch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Возвращает кроп и его бинарную union-маску.
        params:
            idx: индекс элемента
        returns:
            словарь с image (in_ch, H, W) в [-1, 1] и target (1, H, W) из {0, 1}
        """
        sample = super().__getitem__(idx)
        image = np.asarray(sample["image"])
        if image.ndim != 3 or image.shape[2] < self.in_ch:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        target = union_masks(sample["masks"], image.shape[:2])
        x = normalize_hwc(image[..., : self.in_ch])
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            "target": torch.from_numpy(target[None].astype(np.float32)),
        }


def unique_val_items() -> list[tuple]:
    """Возвращает val-элементы без дублей наблюдений.
    returns:
        список кортежей (coco, info, path) по одному на физический снимок
    """
    seen: set[str] = set()
    items = []
    for coco, info, path in _val_items():
        stem = image_stem(info["file_name"])
        if stem in seen:
            continue
        seen.add(stem)
        items.append((coco, info, path))
    return items


def train_items(limit: int = 0, seed: int = DATA_SEED) -> list[tuple]:
    """Возвращает train-элементы без val и без дублей наблюдений.
    params:
        limit: сколько изображений случайно оставить (0 — все)
        seed: сид выбора подмножества
    returns:
        список кортежей (coco, info, path)
    """
    coco = load_coco()
    val = ensure_val_split(coco)
    seen: set[str] = set()
    items = []
    for info in coco.imgs.values():
        stem = image_stem(info["file_name"])
        if stem in val or stem in seen:
            continue
        path = resolve_image(config.train_images, info["file_name"])
        if path is None:
            continue
        seen.add(stem)
        items.append((coco, info, path))
    items.sort(key=lambda item: image_stem(item[1]["file_name"]))
    if limit and len(items) > limit:
        items = random.Random(seed).sample(items, limit)
    return items
