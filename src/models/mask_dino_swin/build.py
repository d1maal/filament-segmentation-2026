"""Сборка Swin-модели, критерия и загрузка чекпоинтов."""

from __future__ import annotations

from pathlib import Path

from src.config import config
from src.models.mask_dino_swin.criterion import SetCriterion
from src.models.mask_dino_swin.matcher import HungarianMatcher
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
    mask_dim: int = 128,
    two_stage: bool = False,
    dn: str = "no",
    dn_num: int = 100,
    noise_scale: float = 0.4,
    deep_supervision: bool = True,
    num_classes: int = 1,
) -> SwinMaskDINO:
    """Собирает SwinMaskDINO с ImageNet-бэкбоном.

    params:
        size: сторона квадратных входов (кратна 32, на архитектуру не влияет)
        backbone: ключ чекпоинта (tiny или small)
        num_queries: число queries
        hidden_dim: ширина трансформера
        dec_layers: число слоёв декодера
        mask_dim: ширина mask-фичей FPN
        two_stage: two-stage выбор queries
        dn: режим denoising ("no" или "seg")
        dn_num: число DN-запросов
        noise_scale: масштаб шума боксов DN
        deep_supervision: aux-лоссы по слоям
        num_classes: число классов
    returns:
        собранная модель
    """
    if backbone not in CHECKPOINTS:
        raise ValueError(
            f"Unknown backbone: {backbone}. Use one of: {', '.join(CHECKPOINTS)}"
        )
    if size % 32:
        raise ValueError(f"size must be divisible by 32, got {size}")
    if size < 224:
        raise ValueError(f"size must be >= 224 (Swin window 7 x stride 32), got {size}")

    def _make(pretrained: bool) -> SwinMaskDINO:
        return SwinMaskDINO(
            CHECKPOINTS[backbone],
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            dec_layers=dec_layers,
            mask_dim=mask_dim,
            two_stage=two_stage,
            dn=dn,
            dn_num=dn_num,
            noise_scale=noise_scale,
            deep_supervision=deep_supervision,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    try:
        return _make(True)
    except Exception as exc:  # noqa: BLE001 — fall back to random init if weights unavailable
        print(f"WARNING: pretrained weights failed ({exc}), using random init")
        return _make(False)


def build_criterion(
    box_loss: bool = True,
    dn: str = "no",
    dec_layers: int = 3,
    two_stage: bool = False,
    num_classes: int = 1,
) -> SetCriterion:
    """Собирает венгерский матчер и критерий лоссов MaskDINO.

    params:
        box_loss: включать боксовые лоссы
        dn: режим denoising
        dec_layers: число слоёв декодера (для aux-весов)
        two_stage: two-stage (для interm-весов)
        num_classes: число классов
    returns:
        критерий с готовым словарём весов
    """
    losses = ["labels", "masks"]
    if box_loss:
        losses.append("boxes")
    weight_dict = {"loss_ce": 4.0, "loss_mask": 5.0, "loss_dice": 5.0}
    if box_loss:
        weight_dict["loss_bbox"] = 5.0
        weight_dict["loss_giou"] = 2.0
    matcher = HungarianMatcher(
        cost_class=4.0,
        cost_mask=1.0,
        cost_dice=1.0,
        cost_box=5.0 if box_loss else 0.0,
        cost_giou=2.0 if box_loss else 0.0,
    )
    if dn == "seg":
        dn_losses = ["labels", "masks"] + (["boxes"] if box_loss else [])
    elif dn == "standard":
        dn_losses = ["labels"] + (["boxes"] if box_loss else [])
    else:
        dn_losses = []
    return SetCriterion(
        num_classes,
        matcher,
        weight_dict,
        losses=tuple(losses),
        dec_layers=dec_layers,
        two_stage=two_stage,
        dn_losses=tuple(dn_losses),
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
        raise SystemExit(
            f"No weights at {weights}. Run training first or pass --weights"
        )
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
        mask_dim=int(checkpoint.get("mask_dim", 128)),
        two_stage=bool(checkpoint.get("two_stage", False)),
        dn=checkpoint.get("dn", "no"),
        dn_num=int(checkpoint.get("dn_num", 100)),
        noise_scale=float(checkpoint.get("noise_scale", 0.4)),
        deep_supervision=bool(checkpoint.get("deep_supervision", True)),
        num_classes=int(checkpoint.get("num_classes", 1)),
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    return weights, checkpoint, model, device
