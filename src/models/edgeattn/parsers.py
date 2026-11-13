from __future__ import annotations

import argparse
from pathlib import Path

from src.models.edgeattn.build import ARCHS


def build_parser() -> argparse.ArgumentParser:    
    p = argparse.ArgumentParser(prog="run.py edgeattn")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="обучение EdgeAttNet или контрольной U-Net")
    t.add_argument("--arch", choices=list(ARCHS), default="edgeattn")
    t.add_argument("--in-ch", dest="in_ch", type=int, choices=(1, 3), default=3)
    t.add_argument("--base", type=int, default=64)
    t.add_argument("--size", type=int, default=512)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--ema", action="store_true")
    t.add_argument("--ema-decay", type=float, default=0.99)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--run-name", type=str, default="edgeattn")
    t.add_argument("--eval-samples-val", type=int, default=0)
    t.add_argument("--eval-samples-train", type=int, default=0)
    t.add_argument("--eval-tiled", action="store_true")
    t.add_argument("--cosine", action="store_true")
    t.add_argument("--warmup-epochs", type=int, default=0)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=0)

    i = sub.add_parser("infer", help="инференс по val и запись submission CSV")
    i.add_argument("--weights", type=Path, default=None)
    i.add_argument("--submit", action="store_true")
    i.add_argument("--score-thr", type=float, default=0.35)
    i.add_argument("--min-area", type=int, default=80)
    i.add_argument("--viz-worst", type=int, default=0)
    i.add_argument("--tiled", action="store_true")
    i.add_argument("--tile-size", type=int, default=512)
    i.add_argument("--tile-overlap", type=int, default=128)
    i.add_argument("--tta", action="store_true")
    i.add_argument("--limit", type=int, default=0)
    return p
