from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.eval.evaluate import cmd_baseline, cmd_check, cmd_infer, cmd_sweep
from src.models.m2f import build_parser, cmd_train


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
            args.max_steps,
            args.accum,
            args.eval_samples_val,
            args.eval_samples_train,
            args.run_name,
            args.augment,
            args.backbone,
            args.object_crop_prob,
            args.negative_crop_prob,
            args.full_image_epochs,
            args.no_object_weight,
            args.dice_weight,
            args.mask_weight,
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


if __name__ == "__main__":
    run_m2f_cli()
