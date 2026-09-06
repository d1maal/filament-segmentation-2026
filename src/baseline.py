from __future__ import annotations

import cv2
import numpy as np

DISK_BLUR_SIGMA = 3.0
DISK_THR_FLOOR = 15.0
DISK_THR_FACTOR = 0.35
DISK_ERODE_KERNEL = 15
INV_BLUR_SIGMA = 1.0
CLOSE_KERNEL = 5
OPEN_KERNEL = 3


def predict_instances(
    image: np.ndarray, min_area: int = 80, max_instances: int = 150
) -> list[np.ndarray]:
    """Выделяет экземпляры волокон на снимке: инвертирует H-alpha, оставляет солнечный диск, применяет порог Otsu и connected components.
    params:
        image: grayscale-изображение (или BGR)
        min_area: минимальная площадь компоненты, попадающая в результат
        max_instances: максимальное число возвращаемых экземпляров
    returns:
        список бинарных масок волокон
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), int(DISK_BLUR_SIGMA))
    disk_bin = (
        blur >= max(DISK_THR_FLOOR, DISK_THR_FACTOR * float(np.median(blur)))
    ).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(disk_bin, 8)
    disk = (
        np.ones_like(gray, np.uint8)
        if n <= 1
        else (labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(
            np.uint8
        )
    )
    disk = cv2.erode(
        disk,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DISK_ERODE_KERNEL, DISK_ERODE_KERNEL)
        ),
    )
    inv = cv2.GaussianBlur(cv2.bitwise_not(gray), (0, 0), INV_BLUR_SIGMA)
    pixels = inv[disk > 0]
    if pixels.size == 0:
        return []
    thr, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((inv >= thr) & (disk > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    labs = [lab for lab in range(1, n) if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area]
    labs.sort(key=lambda lab: int(stats[lab, cv2.CC_STAT_AREA]), reverse=True)
    labs = labs[:max_instances]
    return [(labels == lab).astype(np.uint8) for lab in labs]
