"""Венгерский матчер MaskDINO: сопоставление предсказанных queries с GT-инстансами."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.models.mask_dino_swin.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.models.mask_dino_swin.model import _focal_elementwise


def _mask_cost(
    pred_logits: torch.Tensor, gt_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise focal и dice стоимости масок (Q, G).

    params:
        pred_logits: логиты масок (Q, H, W)
        gt_masks: бинарные маски (G, H, W)
    returns:
        кортеж (focal-стоимость, dice-стоимость), каждая (Q, G)
    """
    q, g = pred_logits.shape[0], gt_masks.shape[0]
    pred = pred_logits.unsqueeze(1).expand(q, g, -1, -1)
    gt = gt_masks.unsqueeze(0).expand(q, g, -1, -1)
    focal = _focal_elementwise(pred, gt).flatten(2).mean(2)
    prob = pred.sigmoid().flatten(2)
    tgt = gt.flatten(2)
    dice = 1 - (2 * (prob * tgt).sum(2) + 1.0) / (prob.sum(2) + tgt.sum(2) + 1.0)
    return focal, dice


class HungarianMatcher(nn.Module):
    """Сопоставляет предсказания и цели минимальной суммарной стоимостью."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_box: float = 1.0,
        cost_giou: float = 1.0,
    ) -> None:
        """Запоминает веса стоимостей.

        params:
            cost_class: вес class-стоимости
            cost_mask: вес focal mask-стоимости
            cost_dice: вес dice-стоимости
            cost_box: вес L1 box-стоимости
            cost_giou: вес GIoU-стоимости
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Считает венгерские пары для каждого изображения батча.

        params:
            outputs: словарь с pred_logits (B, Q, 1), pred_masks (B, Q, H, W), pred_boxes (B, Q, 4)
            targets: список целей с masks (G, H, W) и boxes (G, 4) в cxcywh
        returns:
            список кортежей (индексы строк, индексы столбцов)
        """
        bs, nq = outputs["pred_logits"].shape[:2]
        out_logits = outputs["pred_logits"]
        out_mask = outputs["pred_masks"]
        out_bbox = outputs["pred_boxes"]
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        device = out_logits.device
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(bs):
            tgt_masks = targets[i]["masks"]
            tgt_boxes = targets[i]["boxes"]
            ngt = tgt_masks.shape[0]
            if ngt == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                indices.append((empty, empty))
                continue
            cost_class = _focal_elementwise(
                out_logits[i].expand(nq, ngt),
                torch.ones(nq, ngt, device=device),
            )
            cost_focal, cost_dice = _mask_cost(out_mask[i], tgt_masks)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_box = torch.cdist(out_bbox[i], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(out_bbox_xyxy[i], tgt_boxes_xyxy)
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_focal
                + self.cost_dice * cost_dice
                + self.cost_box * cost_box
                + self.cost_giou * cost_giou
            )
            rows, cols = linear_sum_assignment(C.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(rows, dtype=torch.long, device=device),
                    torch.as_tensor(cols, dtype=torch.long, device=device),
                )
            )
        return indices
