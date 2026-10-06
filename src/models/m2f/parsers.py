from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Строит CLI-парсер с подкомандами check, baseline, train, infer, sweep."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("baseline")
    b.add_argument("--submit", action="store_true")
    b.add_argument("--viz", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--size", type=int, default=512)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--eval-samples-val", type=int, default=0)
    t.add_argument("--eval-samples-train", type=int, default=0)
    t.add_argument("--run-name", type=str, default="m2f")
    t.add_argument(
        "--backbone",
        choices=["tiny", "small", "base", "base-in21k", "large"],
        default="tiny",
    )
    t.add_argument("--full-image-epochs", type=int, default=3)
    t.add_argument("--warmup-epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--min-lr-ratio", type=float, default=0.0)
    t.add_argument("--backbone-lr-mult", type=float, default=0.1)
    t.add_argument("--grad-clip-norm", type=float, default=0.03)
    t.add_argument("--ema-decay", type=float, default=0.996)
    t.add_argument("--grad-checkpoint", action="store_true")
    i = sub.add_parser("infer")
    i.add_argument("--submit", action="store_true")
    i.add_argument("--weights", type=Path, default=None)
    i.add_argument("--score-thr", type=float, default=0.3)
    i.add_argument("--min-area", type=int, default=80)
    i.add_argument("--viz-worst", type=int, default=0)
    i.add_argument("--tiled", action="store_true")
    i.add_argument("--sahi", action="store_true")
    i.add_argument("--tta", action="store_true")
    i.add_argument("--tile-size", type=int, default=1024)
    i.add_argument("--tile-overlap", type=int, default=256)
    i.add_argument("--merge-overlap", type=float, default=0.6)
    i.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("sweep")
    s.add_argument("--weights", type=Path, default=None)
    s.add_argument("--score-thrs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    s.add_argument("--min-areas", type=int, nargs="+", default=[0, 40, 80, 160])
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-log", action="store_true")
    return p
