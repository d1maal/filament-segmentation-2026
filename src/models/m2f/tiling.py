from __future__ import annotations

import numpy as np


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
    """Склеивание перекрывающихся детекций с разных тайлов.
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
                merged[i] = (
                    np.logical_or(mask, kept).astype(np.uint8),
                    max(score, kept_score),
                )
                matched = True
                break
        if not matched:
            merged.append((mask, score))
    return [mask for mask, _ in merged if int(mask.sum()) >= min_area]
