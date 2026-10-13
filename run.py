from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.eval import cmd_baseline, cmd_check, cmd_infer, cmd_sweep
from src.models.m2f import build_parser, cmd_train
from src.models.mask_dino_swin import build_parser as build_mask_dino_swin_parser
from src.models.mask_dino_swin import cmd_infer as cmd_mask_dino_swin_infer
from src.models.mask_dino_swin import cmd_sweep as cmd_mask_dino_swin_sweep
from src.models.mask_dino_swin import cmd_train as cmd_mask_dino_swin_train


def run_m2f_cli(argv: list[str] | None = None) -> None:
    """Разбирает аргументы и вызывает соответствующую подкоманду CLI."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "baseline":
        cmd_baseline(args.submit, args.viz)
    elif args.cmd == "train":
        cmd_train(
            args.smoke,
            args.size,
            args.epochs,
            args.eval_samples_val,
            args.eval_samples_train,
            args.run_name,
            args.backbone,
            args.full_image_epochs,
            args.warmup_epochs,
            args.lr,
            args.backbone_lr_mult,
        )
    elif args.cmd == "infer":
        cmd_infer(
            args.submit,
            args.weights,
            args.score_thr,
            args.min_area,
            args.viz_worst,
            args.tiled,
            args.tile_size,
            args.tile_overlap,
            args.merge_overlap,
            args.limit,
        )
    elif args.cmd == "sweep":
        cmd_sweep(args.weights, args.score_thrs, args.min_areas, args.limit, not args.no_log)
    else:
        p.error(f"unknown command: {args.cmd}")


def run_mask_dino_swin_cli(argv: list[str] | None = None) -> None:
    """Разбирает аргументы Swin-пайплайна и вызывает соответствующую подкоманду."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    p = build_mask_dino_swin_parser()
    args = p.parse_args(argv)
    if args.cmd == "train":
        cmd_mask_dino_swin_train(
            args.smoke,
            args.size,
            args.epochs,
            args.eval_samples_val,
            args.eval_samples_train,
            args.run_name,
            args.backbone,
            args.queries,
            args.lr,
            args.warmup_epochs,
            args.backbone_lr_mult,
        )
    elif args.cmd == "infer":
        cmd_mask_dino_swin_infer(
            args.submit,
            args.weights,
            args.score_thr,
            args.min_area,
            args.limit,
            args.forward_size,
        )
    elif args.cmd == "sweep":
        cmd_mask_dino_swin_sweep(
            args.weights,
            args.score_thrs,
            args.min_areas,
            args.limit,
            not args.no_log,
            args.forward_size,
        )
    else:
        p.error(f"unknown command: {args.cmd}")


def main(argv: list[str] | None = None) -> None:
    """Выбирает пайплайн по первому аргументу: mask_dino_swin или m2f по умолчанию.
    params:
        argv: аргументы командной строки (по умолчанию sys.argv)
    """
    data = sys.argv[1:] if argv is None else argv
    if data and data[0] == "mask_dino_swin":
        run_mask_dino_swin_cli(data[1:])
    else:
        run_m2f_cli(argv)


if __name__ == "__main__":
    main()
