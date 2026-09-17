from __future__ import annotations

import cv2
import numpy as np


def _draw_overlay(image: np.ndarray, gt: list[np.ndarray], pred: list[np.ndarray], metrics: dict) -> np.ndarray:
    """Рисует GT (красное), pred (зелёное) и пересечение (жёлтое).
    params:
        image: grayscale
        gt: истинные маски
        pred: предсказанные маски
        metrics: метрики кадра для подписи
    returns:
        BGR-картинка
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
    gt_mask = np.zeros(image.shape[:2], dtype=bool)
    pred_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask in gt:
        gt_mask |= mask > 0
    for mask in pred:
        pred_mask |= mask > 0
    vis[gt_mask & ~pred_mask] = 0.45 * vis[gt_mask & ~pred_mask] + 0.55 * np.array([0, 0, 255])
    vis[pred_mask & ~gt_mask] = 0.45 * vis[pred_mask & ~gt_mask] + 0.55 * np.array([0, 255, 0])
    vis[gt_mask & pred_mask] = 0.45 * vis[gt_mask & pred_mask] + 0.55 * np.array([0, 255, 255])
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    text = (
        f"PQ {metrics['pq']:.3f} | TP {metrics['tp']} | FP {metrics['fp']} | "
        f"FN {metrics['fn']} | split {metrics['splits']} | merge {metrics['merges']}"
    )
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 980), 36), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return vis