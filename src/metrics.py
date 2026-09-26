from __future__ import annotations

import numpy as np

PQ_IOU_THR = 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Считает метрику Intersection over Union для двух бинарных масок.
    params:
        a: первая бинарная маска
        b: вторая бинарная маска
    returns:
        значение IoU от 0 до 1
    """
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter / union) if union else 0.0


def _iou_matrix(
    gt: list[np.ndarray], pred: list[np.ndarray], chunk: int = 1 << 21
) -> np.ndarray:
    """Матрица IoU всех пар (gt × pred) через одно BLAS-умножение.
    params:
        gt: список истинных масок
        pred: список предсказанных масок
        chunk: размер чанка по пикселям
    returns:
        массив (n_gt, n_pred) со значениями IoU
    """
    n_gt, n_pred = len(gt), len(pred)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    if not n_gt or not n_pred:
        return iou
    g_bin = np.stack([g > 0 for g in gt]).reshape(n_gt, -1)
    p_bin = np.stack([p > 0 for p in pred]).reshape(n_pred, -1)
    area_g = g_bin.sum(axis=1, dtype=np.int64)
    area_p = p_bin.sum(axis=1, dtype=np.int64)
    inter = np.zeros((n_gt, n_pred), dtype=np.float32)
    for start in range(0, g_bin.shape[1], chunk):
        stop = min(g_bin.shape[1], start + chunk)
        g32 = g_bin[:, start:stop].astype(np.float32)
        p32 = p_bin[:, start:stop].astype(np.float32)
        inter += g32 @ p32.T
    union = area_g[:, None] + area_p[None, :] - inter.astype(np.int64)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou


def panoptic_quality(
    gt: list[np.ndarray], pred: list[np.ndarray], thr: float = PQ_IOU_THR
) -> dict:
    """Считает Panoptic Quality между предсказанием и истинной разметкой.
    params:
        gt: список истинных масок
        pred: список предсказанных масок
        thr: порог IoU для совпадения инстансов
    returns:
        словарь с метриками: pq, sq, tp, fp, fn, splits, merges, n_gt, n_pred
    """
    iou = _iou_matrix(gt, pred)
    used_g, used_p, matches = set(), set(), []
    if iou.size:
        for idx in np.argsort(-iou, axis=None):
            gi, pj = map(int, np.unravel_index(int(idx), iou.shape))
            val = float(iou[gi, pj])
            if val < thr:
                break
            if gi in used_g or pj in used_p:
                continue
            used_g.add(gi)
            used_p.add(pj)
            matches.append(val)
    tp, fp, fn = len(matches), len(pred) - len(matches), len(gt) - len(matches)
    iou_sum = sum(matches)
    denom = tp + 0.5 * fp + 0.5 * fn
    splits = (
        int(sum(np.sum(iou[i] >= 0.1) >= 2 for i in range(iou.shape[0])))
        if iou.size
        else 0
    )
    merges = (
        int(sum(np.sum(iou[:, j] >= 0.1) >= 2 for j in range(iou.shape[1])))
        if iou.size
        else 0
    )
    return {
        "pq": float(iou_sum / denom) if denom else 0.0,
        "sq": float(iou_sum / tp) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "splits": splits,
        "merges": merges,
        "n_gt": len(gt),
        "n_pred": len(pred),
    }


def aggregate(rows: list[dict]) -> dict:
    """Собирает общую метрику PQ по всем изображениям, взвешивая по числу экземляров.
    params:
        rows: список словарей метрик от panoptic_quality
    returns:
        словарь с агрегированными pq, tp, fp, fn, splits, merges и числом изображений
    """
    if not rows:
        return {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "splits": 0,
            "merges": 0,
            "images": 0,
        }
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    iou_sum = sum(r["sq"] * r["tp"] for r in rows)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(iou_sum / denom) if denom else 0.0,
        "sq": float(iou_sum / tp) if tp else 0.0,
        "rq": float(tp / denom) if denom else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "splits": sum(r["splits"] for r in rows),
        "merges": sum(r["merges"] for r in rows),
        "images": len(rows),
    }
