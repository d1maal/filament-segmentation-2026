from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Строит CLI-парсер Swin-пайплайна с подкомандами train, infer, sweep."""
    p = argparse.ArgumentParser(prog="run.py mask_dino_swin")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--size", type=int, default=512)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--eval-samples-val", type=int, default=0)
    t.add_argument("--eval-samples-train", type=int, default=0)
    t.add_argument("--run-name", type=str, default="mask_dino_swin")
    t.add_argument("--backbone", choices=["tiny", "small"], default="tiny")
    t.add_argument("--queries", type=int, default=100)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup-epochs", type=int, default=2)
    t.add_argument("--backbone-lr-mult", type=float, default=0.1)
    t.add_argument("--eval-score-thr", type=float, default=0.1)
    t.add_argument("--dec-layers", type=int, default=3)
    t.add_argument("--mask-dim", type=int, default=128)
    t.add_argument("--two-stage", action="store_true")
    t.add_argument("--dn", choices=["no", "standard", "seg"], default="no")
    t.add_argument("--dn-num", type=int, default=100)
    t.add_argument("--noise-scale", type=float, default=0.4)
    t.add_argument("--box-loss", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument(
        "--deep-supervision", action=argparse.BooleanOptionalAction, default=True
    )
    i = sub.add_parser("infer")
    i.add_argument("--submit", action="store_true")
    i.add_argument("--weights", type=Path, default=None)
    i.add_argument("--score-thr", type=float, default=0.3)
    i.add_argument("--min-area", type=int, default=80)
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--forward-size", type=int, default=2048)
    s = sub.add_parser("sweep")
    s.add_argument("--weights", type=Path, default=None)
    s.add_argument(
        "--score-thrs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6]
    )
    s.add_argument("--min-areas", type=int, nargs="+", default=[0, 40, 80, 160])
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-log", action="store_true")
    s.add_argument("--forward-size", type=int, default=2048)
    return p
