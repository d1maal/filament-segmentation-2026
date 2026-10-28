from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.eval import cmd_baseline, cmd_check, cmd_infer, cmd_sweep
from src.models.edgeattn import build_parser as build_edgeattn_parser
from src.models.edgeattn import cmd_infer as edgeattn_cmd_infer
from src.models.edgeattn import cmd_train as edgeattn_cmd_train
from src.models.m2f import build_parser, cmd_train


def run_m2f_cli(argv: list[str] | None = None) -> None:
    """Разбирает аргументы и вызывает соответствующую подкоманду CLI."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "edgeattn":
        edgeattn_args = build_edgeattn_parser().parse_args(args_list[1:])
        if edgeattn_args.cmd == "train":
            edgeattn_cmd_train(
                arch=edgeattn_args.arch,
                in_ch=edgeattn_args.in_ch,
                base=edgeattn_args.base,
                size=edgeattn_args.size,
                epochs=edgeattn_args.epochs,
                batch=edgeattn_args.batch,
                accum=edgeattn_args.accum,
                ema=edgeattn_args.ema,
                ema_decay=edgeattn_args.ema_decay,
                lr=edgeattn_args.lr,
                run_name=edgeattn_args.run_name,
                eval_samples_val=edgeattn_args.eval_samples_val,
                eval_samples_train=edgeattn_args.eval_samples_train,
                eval_tiled=edgeattn_args.eval_tiled,
                cosine=edgeattn_args.cosine,
                workers=edgeattn_args.workers,
                max_steps=edgeattn_args.max_steps,
            )
        elif edgeattn_args.cmd == "infer":
            edgeattn_cmd_infer(
                submit=edgeattn_args.submit,
                weights=edgeattn_args.weights,
                score_thr=edgeattn_args.score_thr,
                min_area=edgeattn_args.min_area,
                viz_worst=edgeattn_args.viz_worst,
                tiled=edgeattn_args.tiled,
                tile_size=edgeattn_args.tile_size,
                tile_overlap=edgeattn_args.tile_overlap,
                limit=edgeattn_args.limit,
                tta=edgeattn_args.tta,
            )
        else:
            raise SystemExit(f"unknown edgeattn command: {edgeattn_args.cmd}")
        return
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
        cmd_sweep(
            args.weights, args.score_thrs, args.min_areas, args.limit, not args.no_log
        )
    else:
        p.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    run_m2f_cli()
