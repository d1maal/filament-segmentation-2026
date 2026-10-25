from __future__ import annotations

import numpy as np


def panoptic_quality_from_labels(
    gt: list[np.ndarray], pred_labels: np.ndarray, thr: float = 0.5
) -> dict:
    """Считает Home PQ по карте меток предсказания без попарных масок.

    Семантика совпадает с src.metrics.panoptic_quality, но пересечения
    считаются через bincount по меткам, что на 2048×2048 даёт ускорение
    примерно в сто раз.
    params:
        gt: список истинных масок инстансов
        pred_labels: карта меток (0 — фон, 1..N — инстансы)
        thr: порог IoU для матчинга
    returns:
        словарь с метриками pq, sq, tp, fp, fn, splits, merges, n_gt, n_pred
    """
    labels = np.asarray(pred_labels)
    n_pred = int(labels.max()) if labels.size else 0
    pred_areas: np.ndarray = np.bincount(labels.reshape(-1), minlength=n_pred + 1)[
        1:
    ].astype(np.int64)
    n_gt = len(gt)
    gt_areas: np.ndarray = np.zeros(n_gt, dtype=np.int64)
    inter: np.ndarray = np.zeros((n_gt, n_pred), dtype=np.int64)
    for i, mask in enumerate(gt):
        binary = np.asarray(mask) > 0
        gt_areas[i] = int(binary.sum())
        if n_pred:
            inter[i] = np.bincount(labels[binary], minlength=n_pred + 1)[1:]
    union = gt_areas[:, None] + pred_areas[None, :] - inter
    iou: np.ndarray = np.zeros((n_gt, n_pred), dtype=np.float64)
    np.divide(inter, union, out=iou, where=union > 0)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[float] = []
    if iou.size:
        for idx in np.argsort(-iou, axis=None):
            gi, pj = map(int, np.unravel_index(int(idx), iou.shape))
            if gi in used_g or pj in used_p:
                continue
            value = float(iou[gi, pj])
            if value < thr:
                break
            used_g.add(gi)
            used_p.add(pj)
            matches.append(value)
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    splits = (
        int(sum(np.sum(iou[i] >= 0.1) >= 2 for i in range(n_gt))) if iou.size else 0
    )
    merges = (
        int(sum(np.sum(iou[:, j] >= 0.1) >= 2 for j in range(n_pred)))
        if iou.size
        else 0
    )
    return {
        "pq": float(sum(matches) / denom) if denom else 0.0,
        "sq": float(sum(matches) / tp) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "splits": splits,
        "merges": merges,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }


def instances_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Превращает карту меток в список бинарных масок инстансов.
    params:
        labels: карта меток (0 — фон, 1..N — инстансы)
    returns:
        список масок uint8
    """
    n = int(labels.max()) if labels.size else 0
    return [(labels == k).astype(np.uint8) for k in range(1, n + 1)]
