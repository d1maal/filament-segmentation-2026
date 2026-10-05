from __future__ import annotations

import numpy as np


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _iou_prepared(a, b, area_a, area_b, bbox_a, bbox_b):
    if area_a == 0 or area_b == 0 or bbox_a is None or bbox_b is None:
        return 0.0
    ay1, ay2, ax1, ax2 = bbox_a
    by1, by2, bx1, bx2 = bbox_b
    y1 = max(ay1, by1)
    y2 = min(ay2, by2)
    x1 = max(ax1, bx1)
    x2 = min(ax2, bx2)
    if y1 >= y2 or x1 >= x2:
        return 0.0
    inter = np.logical_and(a[y1:y2, x1:x2], b[y1:y2, x1:x2]).sum(dtype=np.int64)
    union = area_a + area_b - int(inter)
    return float(inter / union) if union else 0.0


def _prepare_masks(masks):
    binary = [np.asarray(mask) > 0 for mask in masks]
    areas = [int(mask.sum(dtype=np.int64)) for mask in binary]
    bboxes = [_bbox(mask) for mask in binary]
    return binary, areas, bboxes


def panoptic_quality(gt: list[np.ndarray], pred: list[np.ndarray], thr: float = 0.5) -> dict:
    gt_bin, gt_areas, gt_bboxes = _prepare_masks(gt)
    pred_bin, pred_areas, pred_bboxes = _prepare_masks(pred)
    iou = np.zeros((len(gt), len(pred)), dtype=np.float64)
    for i, g in enumerate(gt_bin):
        for j, p in enumerate(pred_bin):
            iou[i, j] = _iou_prepared(g, p, gt_areas[i], pred_areas[j], gt_bboxes[i], pred_bboxes[j])
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
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    iou_sum = sum(matches)
    denom = tp + 0.5 * fp + 0.5 * fn
    splits = int(sum(np.sum(iou[i] >= 0.1) >= 2 for i in range(iou.shape[0]))) if iou.size else 0
    merges = int(sum(np.sum(iou[:, j] >= 0.1) >= 2 for j in range(iou.shape[1]))) if iou.size else 0
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
    if not rows:
        return {"pq": 0.0, "images": 0}
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    iou_sum = sum(r["sq"] * r["tp"] for r in rows)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(iou_sum / denom) if denom else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "splits": sum(r["splits"] for r in rows),
        "merges": sum(r["merges"] for r in rows),
        "images": len(rows),
    }
