from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import aggregate, panoptic_quality

from .config import config
from .dataset import FilamentMaskRCNNDataset
from .infer import load_checkpoint, postprocess_prepared, prepare_output, prepare_tta
from .utils import collate_fn


def _base(score, mask=0.50):
    return {
        "score_threshold": float(score),
        "mask_threshold": float(mask),
        "min_area": 150,
        "deduplicate": True,
        "duplicate_mode": "weighted",
        "mask_nms_iou": 0.60,
        "containment_threshold": 0.90,
        "stability_delta": 0.05,
        "min_mask_confidence": 0.0,
        "min_stability": 0.0,
        "quality_score_threshold": 0.0,
        "mask_confidence_power": 0.50,
        "stability_power": 0.25,
        "enforce_disjoint": True,
        "overlap_score_weight": 0.35,
    }


def _configs():
    configs = []

    for score in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for mask in [0.45, 0.50, 0.55]:
            configs.append(_base(score, mask))

    for score in [0.50, 0.55, 0.60, 0.65]:
        for quality in [0.35, 0.40, 0.45, 0.50]:
            cfg = _base(score)
            cfg["quality_score_threshold"] = quality
            configs.append(cfg)

    for score in [0.50, 0.55, 0.60, 0.65]:
        for stability in [0.45, 0.55, 0.65]:
            cfg = _base(score)
            cfg["min_stability"] = stability
            configs.append(cfg)

    for score in [0.50, 0.55, 0.60, 0.65]:
        for mode in ["suppress", "merge", "weighted"]:
            cfg = _base(score)
            cfg["duplicate_mode"] = mode
            configs.append(cfg)

    for score in [0.50, 0.55, 0.60, 0.65]:
        for nms_iou in [0.50, 0.60, 0.70]:
            cfg = _base(score)
            cfg["mask_nms_iou"] = nms_iou
            configs.append(cfg)

    for score in [0.50, 0.55, 0.60, 0.65]:
        for weight in [0.0, 0.20, 0.35, 0.50, 0.75]:
            cfg = _base(score)
            cfg["overlap_score_weight"] = weight
            configs.append(cfg)

    unique = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


@torch.no_grad()
def run_sweep(
    weights: str | Path,
    limit: int = 0,
    out_csv: str | Path | None = None,
    tta: bool = False,
):
    model, device = load_checkpoint(weights)
    dataset = FilamentMaskRCNNDataset(train=False, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    configs = _configs()
    floor = min(cfg["score_threshold"] for cfg in configs)
    rows_by_config = [[] for _ in configs]

    for image_index, (images, targets) in enumerate(tqdm(loader, desc="sweep val")):
        if limit and image_index >= limit:
            break
        if tta:
            prepared = prepare_tta(
                model,
                images[0],
                device,
                transforms=config.tta_transforms,
                score_floor=floor,
            )
        else:
            output = model([images[0].to(device)])[0]
            prepared = prepare_output(output, score_floor=floor)
        gt = [mask.numpy().astype(np.uint8) for mask in targets[0]["masks"]]

        for cfg_index, cfg in enumerate(configs):
            pred = postprocess_prepared(prepared, **cfg)
            rows_by_config[cfg_index].append(panoptic_quality(gt, pred))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = []
    for cfg, metric_rows in zip(configs, rows_by_config):
        metrics = aggregate(metric_rows)
        results.append({"tta": bool(tta), **cfg, **metrics})
    results.sort(key=lambda row: row["pq"], reverse=True)

    for i, row in enumerate(results[:20], 1):
        print(
            f"{i:02d}. PQ={row['pq']:.5f} score={row['score_threshold']:.2f} "
            f"mask={row['mask_threshold']:.2f} quality={row['quality_score_threshold']:.2f} "
            f"stability={row['min_stability']:.2f} mode={row['duplicate_mode']} "
            f"overlap_w={row['overlap_score_weight']:.2f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"splits={row['splits']} merges={row['merges']}"
        )

    if out_csv is not None and results:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="outputs/mask_rcnn/postprocess_sweep_v4.csv")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    run_sweep(args.weights, limit=args.limit, out_csv=args.out, tta=args.tta)


if __name__ == "__main__":
    main()
