from __future__ import annotations

import torch
from torch.nn import functional as F


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Считает суммарный лосс BCE + Dice, как в EdgeAttNet.
    params:
        logits: логиты модели (B, 1, H, W)
        target: бинарная маска (B, 1, H, W) из {0, 1}
        bce_weight: вес BCE
        dice_weight: вес Dice
        smooth: сглаживание Dice для устойчивости на пустых масках
    returns:
        скалярный тензор лосса
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} != target {tuple(target.shape)}"
        )
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return bce_weight * bce + dice_weight * dice.mean()
