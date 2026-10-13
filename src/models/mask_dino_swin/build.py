"""Сборка Swin-модели и загрузка чекпоинтов."""
from __future__ import annotations

from pathlib import Path

from src.config import config
from src.models.mask_dino_swin.model import SwinMaskDINO

CHECKPOINTS = {
    "tiny": "microsoft/swin-tiny-patch4-window7-224",
    "small": "microsoft/swin-small-patch4-window7-224",
}

OUT_DIR = config.outputs_path / "mask_dino_swin"


def build(
    size: int = 512,
    backbone: str = "tiny",
    num_queries: int = 100,
    hidden_dim: int = 256,
    dec_layers: int = 3,
) -> SwinMaskDINO:
    """Собирает SwinMaskDINO с ImageNet-бэкбоном.
    params:
        size: сторона квадратных входов (кратна 32, на архитектуру не влияет)
        backbone: ключ чекпоинта (tiny или small)
        num_queries: число queries
        hidden_dim: ширина трансформера
        dec_layers: число слоёв декодера
    returns:
        собранная модель
    """
    if backbone not in CHECKPOINTS:
        raise ValueError(f"Unknown backbone: {backbone}. Use one of: {', '.join(CHECKPOINTS)}")
    if size % 32:
        raise ValueError(f"size must be divisible by 32, got {size}")
    try:
        return SwinMaskDINO(
            CHECKPOINTS[backbone],
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            dec_layers=dec_layers,
            pretrained=True,
        )
    except Exception as exc:
        print(f"WARNING: pretrained weights failed ({exc}), using random init")
        return SwinMaskDINO(
            CHECKPOINTS[backbone],
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            dec_layers=dec_layers,
            pretrained=False,
        )


def load_checkpoint(weights: Path | None = None):
    """Загружает чекпоинт и собирает модель с теми же гиперпараметрами, что при train.
    params:
        weights: путь к .pt (по умолчанию outputs/mask_dino_swin/last.pt)
    returns:
        weights, checkpoint, model, device
    """
    import torch

    weights = weights or (OUT_DIR / "last.pt")
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}. Run training first or pass --weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)
    model = build(
        size=int(checkpoint.get("size", 512)),
        backbone=checkpoint.get("backbone", "tiny"),
        num_queries=int(checkpoint.get("num_queries", 100)),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        dec_layers=int(checkpoint.get("dec_layers", 3)),
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    return weights, checkpoint, model, device
