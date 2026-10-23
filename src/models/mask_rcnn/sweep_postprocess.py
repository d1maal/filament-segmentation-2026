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


def _base(score=0.55, mask=0.50, area=150):
    return {
        "score_threshold": float(score),
        "mask_threshold": float(mask),
        "min_area": int(area),
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
        "min_retained_fraction": 0.10,
        "min_tta_votes": 1,
        "single_view_score_threshold": 0.75,
    }


def _configs(tta_mode="none"):
    configs = []
    if tta_mode == "none":
        for score in [0.45, 0.50, 0.55, 0.60, 0.65]:
            for mask in [0.45, 0.50, 0.55]:
                for area in [80, 150, 250]:
                    configs.append(_base(score, mask, area))
        for score in [0.50, 0.55, 0.60]:
            for quality in [0.30, 0.35, 0.40, 0.45]:
                cfg = _base(score)
                cfg["quality_score_threshold"] = quality
                configs.append(cfg)
        for retained in [0.0, 0.10, 0.20, 0.35]:
            for overlap_w in [0.0, 0.20, 0.35, 0.50, 0.75]:
                cfg = _base()
                cfg["min_retained_fraction"] = retained
                cfg["overlap_score_weight"] = overlap_w
                configs.append(cfg)
    else:
        for score in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for votes in [1, 2, 3]:
                for singleton in [0.65, 0.75, 0.85]:
                    cfg = _base(score)
                    cfg["min_tta_votes"] = votes
                    cfg["single_view_score_threshold"] = singleton
                    configs.append(cfg)
        for mask in [0.45, 0.50, 0.55]:
            for area in [80, 150, 250]:
                cfg = _base(0.35, mask, area)
                cfg["min_tta_votes"] = 2
                cfg["single_view_score_threshold"] = 0.75
                configs.append(cfg)
        for retained in [0.0, 0.10, 0.20, 0.35]:
            for overlap_w in [0.0, 0.20, 0.35, 0.50, 0.75]:
                cfg = _base(0.35)
                cfg["min_tta_votes"] = 2
                cfg["single_view_score_threshold"] = 0.75
                cfg["min_retained_fraction"] = retained
                cfg["overlap_score_weight"] = overlap_w
                configs.append(cfg)

    unique = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def _tta_transforms(mode):
    if mode == "light":
        return config.tta_transforms_light
    if mode == "full":
        return config.tta_transforms_full
    return ()


@torch.no_grad()
def run_sweep(
    weights: str | Path,
    limit: int = 0,
    out_csv: str | Path | None = None,
    tta_mode: str = "none",
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
    configs = _configs(tta_mode=tta_mode)
    floor = min(cfg["score_threshold"] for cfg in configs)
    rows_by_config = [[] for _ in configs]
    transforms = _tta_transforms(tta_mode)

    for image_index, (images, targets) in enumerate(tqdm(loader, desc="selector sweep")):
        if limit and image_index >= limit:
            break
        if transforms:
            prepared = prepare_tta(
                model,
                images[0],
                device,
                transforms=transforms,
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
        tp = int(metrics.get("tp", 0))
        fp = int(metrics.get("fp", 0))
        fn = int(metrics.get("fn", 0))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        results.append(
            {
                "tta": tta_mode,
                **cfg,
                **metrics,
                "precision": precision,
                "recall": recall,
            }
        )
    results.sort(key=lambda row: row["pq"], reverse=True)

    for i, row in enumerate(results[:20], 1):
        print(
            f"{i:02d}. PQ={row['pq']:.5f} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"score={row['score_threshold']:.2f} mask={row['mask_threshold']:.2f} "
            f"area={row['min_area']} votes={row['min_tta_votes']} "
            f"singleton={row['single_view_score_threshold']:.2f} "
            f"retained={row['min_retained_fraction']:.2f} overlap_w={row['overlap_score_weight']:.2f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}"
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
    parser.add_argument("--out", default="outputs/mask_rcnn/postprocess_sweep_v5.csv")
    parser.add_argument("--tta", choices=["none", "light", "full"], default="none")
    args = parser.parse_args()
    run_sweep(args.weights, limit=args.limit, out_csv=args.out, tta_mode=args.tta)


if __name__ == "__main__":
    main()
