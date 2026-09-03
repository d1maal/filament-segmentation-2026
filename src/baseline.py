from __future__ import annotations

import cv2
import numpy as np


def predict_instances(image: np.ndarray, min_area: int = 80) -> list[np.ndarray]:
    """Invert H-alpha, keep the disk, Otsu, connected components. Ruler, not a model."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    disk_bin = (blur >= max(15.0, 0.35 * float(np.median(blur)))).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(disk_bin, 8)
    disk = np.ones_like(gray, np.uint8) if n <= 1 else (labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    disk = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    inv = cv2.GaussianBlur(cv2.bitwise_not(gray), (0, 0), 1.0)
    pixels = inv[disk > 0]
    if pixels.size == 0:
        return []
    thr, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((inv >= thr) & (disk > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return [(labels == lab).astype(np.uint8) for lab in range(1, n) if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area]
