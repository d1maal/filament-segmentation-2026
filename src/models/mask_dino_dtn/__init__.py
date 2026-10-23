"""MaskDINO (ResNet-50) — instance segmentation for MAGFiLO.

Vendored from facebookresearch/MaskDINO (Apache-2.0) and adapted for the
project. The third-party module lives in ``maskdino/``; project-specific
integration (dataset registration, mapper, train/infer) is added alongside.
"""

from __future__ import annotations

from src.models.mask_dino_dtn.parsers import build_parser

__all__ = ["build_parser", "cmd_train"]


def __getattr__(name: str):
    """Лениво подгружает detectron2-зависимый атрибут.

    params:
        name: имя запрашиваемого атрибута
    returns:
        запрошенный атрибут
    raises:
        AttributeError: если атрибут неизвестен
    """
    if name == "cmd_train":
        from src.models.mask_dino_dtn.train import cmd_train

        return cmd_train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
