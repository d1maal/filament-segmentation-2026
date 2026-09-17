from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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

DATA_SEED = 42
DATASET_MIN_AREA = 32


class Magfilo(Dataset):
    """Датасет MAGFiLO из train-изображений."""

    def __init__(
        self,
        size: int = 512,
        min_area: int = DATASET_MIN_AREA,
        object_crop_prob: float = 0.7,
        negative_crop_prob: float = 0.1,
        augment: bool = True,
    ) -> None:
        """Составляет список train-изображений, исключая валидационный сплит.
        params:
            size: сторона кропа
            min_area: минимальная площадь маски в кропе
            object_crop_prob: доля кропов вокруг филамента
            negative_crop_prob: доля пустых кропов
            augment: геометрические аугментации (flip/повороты)
        """
        if object_crop_prob < 0 or negative_crop_prob < 0 or object_crop_prob + negative_crop_prob > 1:
            raise ValueError("Crop probabilities must be >= 0 and sum to <= 1")
        self.size = size
        self.min_area = min_area
        self.object_crop_prob = object_crop_prob
        self.negative_crop_prob = negative_crop_prob
        self.augment = augment
        self.full_image = False
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

    def set_full_image(self, enabled: bool) -> None:
        """Включает или выключает обучение на полном диске вместо кропа.
        params:
            enabled: True — без кропа
        """
        self.full_image = enabled

    def _random_xy(self, h: int, w: int) -> tuple[int, int]:
        """Случайный левый верхний угол кропа.
        params:
            h: высота изображения
            w: ширина изображения
        returns:
            координаты (y, x)
        """
        y = int(self.rng.randint(0, h - self.size + 1))
        x = int(self.rng.randint(0, w - self.size + 1))
        return y, x

    def _crop(
        self, image: np.ndarray, masks: list[np.ndarray], y: int, x: int
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Вырезает квадрат и оставляет маски с площадью не меньше min_area.
        params:
            image: изображение
            masks: маски инстансов
            y: верх кропа
            x: левый край кропа
        returns:
            кроп и отфильтрованные маски
        """
        crop = image[y : y + self.size, x : x + self.size]
        cropped = [m[y : y + self.size, x : x + self.size] for m in masks]
        cropped = [m for m in cropped if int(m.sum()) >= self.min_area]
        return crop, cropped

    def _object_crop(
        self, image: np.ndarray, masks: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Кроп вокруг случайного филамента с небольшим сдвигом.
        params:
            image: изображение
            masks: маски инстансов
        returns:
            кроп и маски
        """
        h, w = image.shape[:2]
        valid = [m for m in masks if int(m.sum()) >= self.min_area]
        if not valid:
            y, x = self._random_xy(h, w)
            return self._crop(image, masks, y, x)
        target = valid[int(self.rng.randint(0, len(valid)))]
        ys, xs = np.where(target > 0)
        jitter = max(1, self.size // 8)
        center_y = (int(ys.min()) + int(ys.max())) // 2 + int(self.rng.randint(-jitter, jitter + 1))
        center_x = (int(xs.min()) + int(xs.max())) // 2 + int(self.rng.randint(-jitter, jitter + 1))
        y = int(np.clip(center_y - self.size // 2, 0, h - self.size))
        x = int(np.clip(center_x - self.size // 2, 0, w - self.size))
        return self._crop(image, masks, y, x)

    def _negative_crop(
        self, image: np.ndarray, masks: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Пытается взять кроп без филаментов.
        params:
            image: изображение
            masks: маски инстансов
        returns:
            кроп и маски (пустой список, если получилось)
        """
        h, w = image.shape[:2]
        fallback = None
        for _ in range(16):
            y, x = self._random_xy(h, w)
            crop, cropped = self._crop(image, masks, y, x)
            fallback = (crop, cropped)
            if not cropped:
                return crop, []
        return fallback

    def _augment(
        self, image: np.ndarray, masks: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Синхронно отражает и поворачивает изображение и маски.
        params:
            image: изображение
            masks: маски инстансов
        returns:
            преобразованные image и masks
        """
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