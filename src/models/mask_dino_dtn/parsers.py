"""CLI-парсер для модели MaskDINO (ResNet-50)."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "magfilo_R50_3ch.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Строит парсер с подкомандой train.

    returns:
        парсер; подкоманды infer/sweep добавятся позже
    """
    p = argparse.ArgumentParser(prog="mask_dino_dtn")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--size", type=int, default=512)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--run-name", type=str, default="mask_dino_dtn")
    t.add_argument("--ims-per-batch", type=int, default=None)
    t.add_argument("--num-workers", type=int, default=None)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    t.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return p
