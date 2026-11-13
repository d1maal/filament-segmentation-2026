from __future__ import annotations

from pathlib import Path

from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from src.config import config

CHECKPOINTS = {
    "tiny": "facebook/mask2former-swin-tiny-coco-instance",
    "small": "facebook/mask2former-swin-small-coco-instance",
}


def build(
    size: int = 512,
    backbone: str = "tiny",
    no_object_weight: float = 0.1,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
):
    """Сборка процессора и модели Mask2Former для сегментации филаментов.
    params:
        size: сторона входных квадратных изображений
        backbone: ключ чекпоинта (tiny или small)
        no_object_weight: вес класса «нет объекта»
        dice_weight: вес Dice-лосса
        mask_weight: вес mask-лосса
    returns:
        кортеж из процессора и модели
    """
    if backbone not in CHECKPOINTS:
        raise ValueError(
            f"Unknown backbone: {backbone}. Use one of: {', '.join(CHECKPOINTS)}"
        )
    ckpt = CHECKPOINTS[backbone]
    processor = Mask2FormerImageProcessor.from_pretrained(
        ckpt, size={"height": size, "width": size}
    )
    model_cfg = Mask2FormerConfig.from_pretrained(
        ckpt,
        id2label={0: "filament"},
        label2id={"filament": 0},
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ckpt, config=model_cfg, ignore_mismatched_sizes=True
    )
    return processor, model


def load_checkpoint(weights: Path | None = None):
    """Загрузка чекпоинта и сборка модели с теми же гиперпараметрами, что при train.
    params:
        weights: путь к .pt (по умолчанию last.pt)
    returns:
        weights, checkpoint, processor, model, device
    """
    import torch

    weights = weights or (config.models_path / "last.pt")
    if not weights.exists():
        raise SystemExit(
            f"No weights at {weights}. Run training first or pass --weights"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)
    processor, model = build(
        int(checkpoint.get("size", 512)),
        backbone=checkpoint.get("backbone", "tiny"),
        no_object_weight=float(checkpoint.get("no_object_weight", 0.1)),
        dice_weight=float(checkpoint.get("dice_weight", 5.0)),
        mask_weight=float(checkpoint.get("mask_weight", 5.0)),
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    return weights, checkpoint, processor, model, device
