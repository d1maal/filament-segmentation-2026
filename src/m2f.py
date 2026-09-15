from __future__ import annotations

import csv
import math
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
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

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

CHECKPOINTS = {
    "tiny": "facebook/mask2former-swin-tiny-coco-instance",
    "small": "facebook/mask2former-swin-small-coco-instance",
}
DATA_SEED = 42
DATASET_MIN_AREA = 32
BEST_WORST_TOP_K = 10


def build(
    size: int = 512,
    backbone: str = "tiny",
    no_object_weight: float = 0.1,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
):
    """Собирает процессор и модель Mask2Former для сегментации филаментов.
    params:
        size: сторона входных квадратных изображений
        backbone: ключ чекпоинта (tiny или small)
        no_object_weight: вес класса «нет объекта»
        dice_weight: вес Dice-лосса
        mask_weight: вес mask-лосса
    returns:
        кортеж из процессора и модели
    """
    if backbone not in CHECKPOINTS:
        raise ValueError(f"Unknown backbone: {backbone}. Use one of: {', '.join(CHECKPOINTS)}")
    ckpt = CHECKPOINTS[backbone]
    processor = Mask2FormerImageProcessor.from_pretrained(
        ckpt, size={"height": size, "width": size}
    )
    model_cfg = Mask2FormerConfig.from_pretrained(
        ckpt,
        id2label={0: "filament"},
        label2id={"filament": 0},
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ckpt, config=model_cfg, ignore_mismatched_sizes=True
    )
    return processor, model


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


def _instances_from_result(result: dict, score_thr: float, min_area: int) -> list[tuple[np.ndarray, float]]:
    """Достаёт маски и скоры из выхода post_process_instance_segmentation.
    params:
        result: словарь сегментации процессора
        score_thr: минимальный скор
        min_area: минимальная площадь маски
    returns:
        список пар (маска, скор)
    """
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


@torch.no_grad()
def _predict_scored(model, processor, image: np.ndarray, device, score_thr: float, min_area: int = 0):
    """Инференс одного изображения со скорами.
    params:
        model: модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale или BGR
        device: устройство
        score_thr: минимальный скор
        min_area: минимальная площадь
    returns:
        список пар (маска, скор)
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
    return _instances_from_result(result, score_thr, min_area)


@torch.no_grad()
def predict_masks(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float = 0.3,
    min_area: int = 80,
) -> list[np.ndarray]:
    """Делает инференс модели на одном полном изображении.
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
    return [mask for mask, _ in _predict_scored(model, processor, image, device, score_thr, min_area)]


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Стартовые координаты тайлов вдоль одной оси.
    params:
        length: длина оси
        tile_size: размер тайла
        overlap: перекрытие соседних тайлов
    returns:
        список стартов
    """
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    end = length - tile_size
    if starts[-1] != end:
        starts.append(end)
    return starts


def _mask_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Сходство двух масок: max(IoU, containment).
    params:
        a: первая маска
        b: вторая маска
    returns:
        число от 0 до 1
    """
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
    """Склеивает перекрывающиеся детекции с разных тайлов.
    params:
        candidates: пары (маска, скор)
        overlap_thr: порог сходства для слияния
        min_area: минимальная площадь после слияния
    returns:
        список масок
    """
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
    """Инференс полным кадром плюс перекрывающимися тайлами.
    params:
        model: модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale или BGR
        device: устройство
        score_thr: минимальный скор
        min_area: минимальная площадь
        tile_size: сторона тайла
        overlap: перекрытие тайлов
        merge_overlap: порог слияния дублей
        include_full: добавить предсказание по целому кадру
    returns:
        список масок
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    candidates: list[tuple[np.ndarray, float]] = []
    if include_full:
        candidates.extend(_predict_scored(model, processor, gray, device, score_thr, 0))
    margin = overlap // 2
    for y in _tile_starts(h, tile_size, overlap):
        for x in _tile_starts(w, tile_size, overlap):
            tile = gray[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            tile_h, tile_w = tile.shape[:2]
            for mask, score in _predict_scored(model, processor, tile, device, score_thr, 0):
                ys_mask, xs_mask = np.where(mask > 0)
                if not len(xs_mask):
                    continue
                center_y = float(ys_mask.mean())
                center_x = float(xs_mask.mean())
                top = 0 if y == 0 else margin
                left = 0 if x == 0 else margin
                bottom = 0 if y + tile_h >= h else margin
                right = 0 if x + tile_w >= w else margin
                if not (top <= center_y < tile_h - bottom and left <= center_x < tile_w - right):
                    continue
                full = np.zeros((h, w), dtype=np.uint8)
                full[y : y + tile_h, x : x + tile_w] = mask
                candidates.append((full, score))
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
    """Один forward, несколько порогов score/min_area — для sweep.
    params:
        model: модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale или BGR
        device: устройство
        score_thrs: пороги скора
        min_areas: пороги площади
    returns:
        словарь (score_thr, min_area) → маски
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inputs = {
        k: v.to(device)
        for k, v in processor(images=three_channel(gray), return_tensors="pt").items()
    }
    outputs = model(**inputs)
    predictions = {}
    for score_thr in score_thrs:
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(score_thr),
            target_sizes=[(h, w)],
            return_binary_maps=True,
        )[0]
        masks = [mask for mask, _ in _instances_from_result(result, float(score_thr), 0)]
        areas = [int(mask.sum()) for mask in masks]
        for min_area in min_areas:
            predictions[(float(score_thr), int(min_area))] = [
                mask for mask, area in zip(masks, areas) if area >= int(min_area)
            ]
    return predictions


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
    items.sort(key=lambda item: image_stem(item[1]["file_name"]))
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
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
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
        eval_samples: число случайных изображений для оценки (0 — не считать)
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
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
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
        results.append((pq, info, path, coco))
    results.sort(key=lambda r: r[0])
    for name, seq in (("bad", results[:BEST_WORST_TOP_K]), ("good", results[-BEST_WORST_TOP_K:])):
        folder = run_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for pq, info, path, coco in seq:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for m in instances_for_image(coco, info["id"]):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
            for m in predict_masks(model, processor, gray, device):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 255, 0), 2)
            cv2.imwrite(str(folder / f"pq{pq:.3f}_{path.stem}.png"), vis)


def _make_scheduler(optimizer, updates_per_epoch: int, epochs: int, warmup_epochs: int):
    """Warmup и cosine decay по шагам оптимизатора.
    params:
        optimizer: AdamW
        updates_per_epoch: число step() за эпоху
        epochs: число эпох
        warmup_epochs: число эпох линейного разогрева
    returns:
        LambdaLR
    """
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
        valid_val = [(e, v) for e, v in zip(epochs, history["val_pq"]) if v is not None]
        valid_train = [(e, v) for e, v in zip(epochs, history["train_pq"]) if v is not None]
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
    augment: bool = True,
    backbone: str = "tiny",
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    full_image_epochs: int = 3,
    no_object_weight: float = 0.08,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
    warmup_epochs: int = 2,
    backbone_lr_mult: float = 0.1,
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
        augment: геометрические аугментации
        backbone: tiny или small
        object_crop_prob: доля кропов вокруг филамента
        negative_crop_prob: доля пустых кропов
        full_image_epochs: сколько последних эпох учить на полном диске
        no_object_weight: вес класса «нет объекта»
        dice_weight: вес Dice-лосса
        mask_weight: вес mask-лосса
        warmup_epochs: эпохи линейного разогрева lr
        backbone_lr_mult: множитель lr для энкодера
    returns:
        словарь с историей train_loss, train_pq и val_pq по эпохам
    """
    if not config.train_annotations.exists():
        raise SystemExit(
            f"No data at {config.train_annotations}. On Kaggle: Add Data → filament-segmentation-2026"
        )
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
        augment=augment,
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
    scheduler = _make_scheduler(
        opt, max(1, math.ceil(len(loader) / max(1, accum))), epochs, warmup_epochs
    )
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
    extra = {
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
        "augment": augment,
    }
    metrics_rows: list[dict] = []
    best_pq = -1.0
    history: dict = {"train_loss": [], "val_pq": [], "train_pq": []}
    run = None
    if init_dagshub():
        try:
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
                    "checkpoint": CHECKPOINTS[backbone],
                    "size": size,
                    "epochs": epochs,
                    "accum": accum,
                    "weight_decay": 0.05,
                    "workers": workers,
                    **extra,
                }
            )
        except Exception as exc:
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None
    step_i = 0
    ckpt = out / "last.pt"
    for epoch in range(1, epochs + 1):
        use_full = full_image_epochs > 0 and epoch > epochs - full_image_epochs
        data.set_full_image(use_full)
        mode = "full" if use_full else "crop"
        running, seen, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch} [{mode}]"), start=1):
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
        bundle = {"model": model.state_dict(), "size": size, "epoch": epoch, **extra}
        torch.save(bundle, ckpt)
        torch.save(bundle, weights_dir / f"epoch_{epoch}.pt")
        avg_loss = running / max(seen, 1)
        history["train_loss"].append(avg_loss)
        model.eval()
        val_pq = _eval_val(model, processor, device, eval_val_samples)
        train_pq = (
            _eval_train(model, processor, device, eval_train_samples) if eval_train_samples else None
        )
        model.train()
        history["val_pq"].append(val_pq)
        history["train_pq"].append(train_pq)
        if run is not None:
            metrics = {"train_loss": avg_loss, "val_pq": val_pq}
            if train_pq is not None:
                metrics["train_pq"] = train_pq
            mlflow.log_metrics(metrics, step=epoch)
        metrics_rows.append(
            {"epoch": epoch, "train_loss": avg_loss, "train_pq": train_pq, "val_pq": val_pq}
        )
        if val_pq > best_pq:
            best_pq = val_pq
            torch.save(bundle, run_dir / "best.pt")
        print(
            f"epoch {epoch} mode={mode} loss={avg_loss:.4f} "
            f"lr={opt.param_groups[1]['lr']:.2e} -> {ckpt}"
        )
        if max_steps and step_i >= max_steps:
            break
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_pq", "val_pq"])
        w.writeheader()
        w.writerows(metrics_rows)
    processor.save_pretrained(out / "processor")
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=False)["model"]
        )
        model.eval()
        _save_best_worst(model, processor, device, run_dir)
    if run is not None:
        _log_training_artifacts(model, ckpt, history)
        mlflow.end_run()
    return history


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
    augment: bool = True,
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
    """Обучает Mask2Former на MAGFiLO через log_train и возвращает путь к last.pt.
    params:
        те же, что у log_train
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
        backbone=backbone,
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        full_image_epochs=full_image_epochs,
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
        warmup_epochs=warmup_epochs,
        backbone_lr_mult=backbone_lr_mult,
    )
    return config.models_path / "last.pt"
