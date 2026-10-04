from __future__ import annotations

import cv2
import numpy as np
import torch

from src.data import three_channel
from src.models.m2f.tiling import _merge_candidates, _tile_starts


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
def predict_masks_tta(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float = 0.3,
    min_area: int = 80,
    merge_overlap: float = 0.6,
) -> list[np.ndarray]:
    """Инференс с test-time augmentation (повороты и отражения) и слиянием масок.
    params:
        model: модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale или BGR
        device: устройство
        score_thr: минимальный скор
        min_area: минимальная площадь после слияния
        merge_overlap: порог сходства для слияния дублей
    returns:
        список масок
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    views: list[tuple[str, np.ndarray]] = [("", gray)]
    views += [(f"r{k}", np.rot90(gray, k)) for k in (1, 2, 3)]
    flipped = np.flip(gray, axis=1)
    views += [(f"f{k}", np.rot90(flipped, k)) for k in (0, 1, 2, 3)]
    candidates: list[tuple[np.ndarray, float]] = []
    for name, view in views:
        for mask, score in _predict_scored(model, processor, view, device, score_thr, 0):
            if name == "":
                back = mask
            elif name[0] == "r":
                back = np.rot90(mask, -int(name[1:]))
            else:
                back = np.flip(np.rot90(mask, -int(name[1:])), axis=1)
            candidates.append((back, score))
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


def _predict(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float,
    min_area: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    merge_overlap: float,
    sahi: bool = False,
    tta: bool = False,
):
    """Выбирает полный инференс, тайловый, SAHI или TTA.
    params:
        model: модель Mask2Former
        processor: процессор
        image: grayscale
        device: устройство
        score_thr: порог скора
        min_area: минимальная площадь
        tiled: использовать тайлы
        tile_size: сторона тайла
        tile_overlap: перекрытие
        merge_overlap: порог слияния дублей
        sahi: использовать SAHI sliced-инференс
        tta: использовать test-time augmentation
    returns:
        список масок
    """
    if tta:
        return predict_masks_tta(
            model,
            processor,
            image,
            device,
            score_thr=score_thr,
            min_area=min_area,
            merge_overlap=merge_overlap,
        )
    if sahi:
        from src.models.m2f.sahi import predict_masks_sahi

        return predict_masks_sahi(
            model,
            processor,
            image,
            device,
            score_thr=score_thr,
            min_area=min_area,
            slice_size=tile_size,
            overlap=tile_overlap,
            merge_overlap=merge_overlap,
        )
    if tiled:
        return predict_masks_tiled(
            model,
            processor,
            image,
            device,
            score_thr=score_thr,
            min_area=min_area,
            tile_size=tile_size,
            overlap=tile_overlap,
            merge_overlap=merge_overlap,
        )
    return predict_masks(model, processor, image, device, score_thr=score_thr, min_area=min_area)