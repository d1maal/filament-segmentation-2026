"""Утилиты для работы с боксами: преобразование форматов, GIoU, маски → боксы."""

from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Считает площади боксов в формате (x0, y0, x1, y1).

    params:
        boxes: тензор (..., 4) в формате xyxy
    returns:
        площади (...,)
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Преобразует боксы из (x0, y0, x1, y1) в (cx, cy, w, h).

    params:
        x: тензор (..., 4) в формате xyxy
    returns:
        тензор (..., 4) в формате cxcywh
    """
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Преобразует боксы из (cx, cy, w, h) в (x0, y0, x1, y1).

    params:
        x: тензор (..., 4) в формате cxcywh
    returns:
        тензор (..., 4) в формате xyxy
    """
    xc, yc, w, h = x.unbind(-1)
    return torch.stack([xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise Generalized IoU между двумя наборами боксов.

    params:
        boxes1: тензор (N, 4) в формате xyxy
        boxes2: тензор (M, 4) в формате xyxy
    returns:
        матрица GIoU (N, M)
    """
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / (area + 1e-6)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Считает ограничивающие боксы (xyxy) вокруг бинарных масок.

    params:
        masks: тензор (N, H, W) бинарных масок
    returns:
        тензор (N, 4) в формате xyxy; для пустого входа — (0, 4)
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)
    h, w = masks.shape[-2:]
    y = torch.arange(h, dtype=torch.float32, device=masks.device)
    x = torch.arange(w, dtype=torch.float32, device=masks.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    bool_m = masks > 0.5
    xs = xx.unsqueeze(0).expand(masks.shape[0], -1, -1)
    ys = yy.unsqueeze(0).expand(masks.shape[0], -1, -1)
    x_min = xs.masked_fill(~bool_m, 1e8).flatten(1).min(1).values
    x_max = (xs * bool_m).flatten(1).max(1).values
    y_min = ys.masked_fill(~bool_m, 1e8).flatten(1).min(1).values
    y_max = (ys * bool_m).flatten(1).max(1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def masks_to_boxes_normalized(masks: torch.Tensor) -> torch.Tensor:
    """Боксы cxcywh, нормализованные по размеру маски.

    params:
        masks: бинарные маски (N, H, W)
    returns:
        боксы (N, 4) в формате cxcywh в [0, 1]
    """
    xyxy = masks_to_boxes(masks)
    h, w = masks.shape[-2:]
    scale = torch.tensor([w, h], dtype=torch.float32, device=masks.device)
    cxcy = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    wh = xyxy[:, 2:] - xyxy[:, :2]
    return torch.cat([cxcy / scale, wh / scale], dim=-1)
