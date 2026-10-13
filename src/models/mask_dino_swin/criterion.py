"""SetCriterion MaskDINO: bipartite-матчинг и лоссы (labels/masks/boxes + aux + interm + DN)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mask_dino_swin.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.models.mask_dino_swin.matcher import HungarianMatcher
from src.models.mask_dino_swin.model import (
    _focal_elementwise,
    dice_loss,
    sigmoid_focal_loss,
)


def _get_src_permutation_idx(indices: list[tuple[torch.Tensor, torch.Tensor]]):
    """Перестановочные индексы строк (queries) для собранных пар."""
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices: list[tuple[torch.Tensor, torch.Tensor]]):
    """Перестановочные индексы столбцов (GT) для собранных пар."""
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
    )
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class SetCriterion(nn.Module):
    """Считает лоссы MaskDINO по результатам венгерского матчинга."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        losses: tuple[str, ...] = ("labels", "masks"),
        dec_layers: int = 3,
        two_stage: bool = False,
        dn_losses: tuple[str, ...] = (),
        eos_coef: float = 0.1,
    ) -> None:
        """Запоминает конфигурацию лоссов и строит полный словарь весов.

        params:
            num_classes: число классов
            matcher: венгерский матчер
            weight_dict: базовые веса лоссов
            losses: типы лоссов главной ветки
            dec_layers: число слоёв декодера (для aux-весов)
            two_stage: добавлять interm-веса
            dn_losses: типы DN-лоссов
            eos_coef: вес «нет объекта»
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.dn_losses = dn_losses
        self.eos_coef = eos_coef
        self.weight_dict = dict(weight_dict)
        if two_stage:
            self.weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})
        for i in range(dec_layers - 1):
            self.weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        for k, v in weight_dict.items():
            self.weight_dict[k + "_dn"] = v

    def loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """Focal loss классификации (бинарный «есть объект»).

        params:
            outputs: предсказания
            targets: цели
            indices: венгерские пары
            num_boxes: нормировка
        returns:
            словарь с loss_ce
        """
        src_logits = outputs["pred_logits"]
        idx = _get_src_permutation_idx(indices)
        target = torch.zeros_like(src_logits)
        target[idx] = 1.0
        el = _focal_elementwise(src_logits, target)
        weights = torch.where(
            target > 0, torch.ones_like(el), self.eos_coef * torch.ones_like(el)
        )
        return {"loss_ce": (el * weights).sum() / num_boxes}

    def loss_masks(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """Focal + dice лоссы масок на сопоставленных парах.

        params:
            outputs: предсказания
            targets: цели
            indices: венгерские пары
            num_boxes: нормировка
        returns:
            словарь с loss_mask и loss_dice
        """
        src_masks = outputs["pred_masks"]
        src_idx = _get_src_permutation_idx(indices)
        matched = src_masks[src_idx]
        target_masks = torch.cat(
            [t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        if matched.numel() == 0:
            device = src_masks.device
            return {
                "loss_mask": torch.tensor(0.0, device=device),
                "loss_dice": torch.tensor(0.0, device=device),
            }
        return {
            "loss_mask": sigmoid_focal_loss(matched, target_masks),
            "loss_dice": dice_loss(matched, target_masks),
        }

    def loss_boxes(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """L1 + GIoU лоссы боксов на сопоставленных парах.

        params:
            outputs: предсказания
            targets: цели
            indices: венгерские пары
            num_boxes: нормировка
        returns:
            словарь с loss_bbox и loss_giou
        """
        src_boxes = outputs["pred_boxes"]
        src_idx = _get_src_permutation_idx(indices)
        matched = src_boxes[src_idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        if matched.numel() == 0:
            device = src_boxes.device
            return {
                "loss_bbox": torch.tensor(0.0, device=device),
                "loss_giou": torch.tensor(0.0, device=device),
            }
        loss_bbox = F.l1_loss(matched, target_boxes, reduction="none").sum() / num_boxes
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(matched), box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def get_loss(
        self,
        loss: str,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """Диспетчер лоссов по имени.

        params:
            loss: имя лосса ("labels", "masks", "boxes")
            outputs: предсказания
            targets: цели
            indices: венгерские пары
            num_boxes: нормировка
        returns:
            словарь лоссов
        """
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "boxes": self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def _dn_losses(self, dn_meta: dict, num_boxes: float) -> dict[str, torch.Tensor]:
        """Лоссы для denoising-ветки (по известным индексам, без матчинга).

        params:
            dn_meta: метаданные DN
            num_boxes: нормировка
        returns:
            словарь DN-лоссов
        """
        out = dn_meta["output_known"]
        known_boxes = dn_meta["known_boxes"]
        known_masks = dn_meta["known_masks"]
        bid = dn_meta["known_bid"]
        qidx = dn_meta["map_known_indice"]
        pred_logits = out["pred_logits"][bid, qidx]
        pred_boxes = out["pred_boxes"][bid, qidx]
        pred_masks = out["pred_masks"][bid, qidx]
        h, w = pred_masks.shape[-2:]
        if known_masks.shape[-2:] != (h, w):
            known_masks = F.interpolate(
                known_masks.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).squeeze(1)
        losses: dict[str, torch.Tensor] = {}
        if "labels" in self.dn_losses:
            losses["loss_ce_dn"] = (
                _focal_elementwise(pred_logits, torch.ones_like(pred_logits)).sum()
                / num_boxes
            )
        if "masks" in self.dn_losses:
            losses["loss_mask_dn"] = sigmoid_focal_loss(pred_masks, known_masks)
            losses["loss_dice_dn"] = dice_loss(pred_masks, known_masks)
        if "boxes" in self.dn_losses:
            losses["loss_bbox_dn"] = (
                F.l1_loss(pred_boxes, known_boxes, reduction="none").sum() / num_boxes
            )
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(known_boxes)
            )
            losses["loss_giou_dn"] = (1 - torch.diag(giou)).sum() / num_boxes
        return losses

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        """Считает все лоссы: главные, aux, interm и DN.

        params:
            outputs: словарь предсказаний модели
            targets: список целей
        returns:
            словарь лоссов (без весов)
        """
        src_masks = outputs["pred_masks"]
        h, w = src_masks.shape[-2:]
        resized_targets = []
        for t in targets:
            m = t["masks"]
            if m.numel() and m.shape[-2:] != (h, w):
                m = F.interpolate(
                    m.unsqueeze(1).float(), size=(h, w), mode="nearest"
                ).squeeze(1)
            resized_targets.append({**t, "masks": m})
        targets = resized_targets

        main = {
            k: v
            for k, v in outputs.items()
            if k not in ("aux_outputs", "interm_outputs", "dn_meta")
        }
        indices = self.matcher(main, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        losses: dict[str, torch.Tensor] = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, main, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux, targets, aux_indices, num_boxes)
                    losses.update({k + f"_{i}": v for k, v in l_dict.items()})

        interm = outputs.get("interm_outputs")
        if interm is not None:
            interm_indices = self.matcher(interm, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm, targets, interm_indices, num_boxes)
                losses.update({k + "_interm": v for k, v in l_dict.items()})

        dn_meta = outputs.get("dn_meta")
        if dn_meta is not None:
            losses.update(self._dn_losses(dn_meta, num_boxes))

        return losses
