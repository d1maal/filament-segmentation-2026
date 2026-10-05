from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import aggregate, panoptic_quality

from .dataset import FilamentMaskRCNNDataset
from .infer import load_checkpoint, postprocess_prepared, prepare_output
from .utils import collate_fn


def _configs():
    configs = []
    for score in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for mask_threshold in [0.45, 0.50, 0.55]:
            configs.append({
                "score_threshold": score,
                "mask_threshold": mask_threshold,
                "min_area": 150,
                "deduplicate": True,
                "duplicate_mode": "merge",
                "mask_nms_iou": 0.60,
                "containment_threshold": 0.90,
                "enforce_disjoint": True,
                "overlap_score_power": 0.50,
            })
    for score in [0.50, 0.55, 0.60, 0.65]:
        for area in [100, 250]:
            configs.append({
                "score_threshold": score,
                "mask_threshold": 0.50,
                "min_area": area,
                "deduplicate": True,
                "duplicate_mode": "merge",
                "mask_nms_iou": 0.60,
                "containment_threshold": 0.90,
                "enforce_disjoint": True,
                "overlap_score_power": 0.50,
            })
    for score in [0.50, 0.55, 0.60, 0.65]:
        for mode in ["none", "suppress", "merge"]:
            for iou in [0.55, 0.65]:
                configs.append({
                    "score_threshold": score,
                    "mask_threshold": 0.50,
                    "min_area": 150,
                    "deduplicate": mode != "none",
                    "duplicate_mode": mode,
                    "mask_nms_iou": iou,
                    "containment_threshold": 0.90,
                    "enforce_disjoint": True,
                    "overlap_score_power": 0.50,
                })
    for score in [0.50, 0.55, 0.60, 0.65]:
        for power in [0.0, 1.0]:
            configs.append({
                "score_threshold": score,
                "mask_threshold": 0.50,
                "min_area": 150,
                "deduplicate": True,
                "duplicate_mode": "merge",
                "mask_nms_iou": 0.60,
                "containment_threshold": 0.90,
                "enforce_disjoint": True,
                "overlap_score_power": power,
            })
    unique = []
    seen = set()
    for cfg in configs:
        key = tuple(cfg.items())
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


@torch.no_grad()
def run_sweep(weights: str | Path, limit: int = 0, out_csv: str | Path | None = None):
    model, device = load_checkpoint(weights)
    dataset = FilamentMaskRCNNDataset(train=False, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    configs = _configs()
    floor = min(cfg["score_threshold"] for cfg in configs)
    rows_by_config = [[] for _ in configs]
    for image_index, (images, targets) in enumerate(tqdm(loader, desc="sweep val")):
        if limit and image_index >= limit:
            break
        output = model([images[0].to(device)])[0]
        prepared = prepare_output(output, score_floor=floor)
        gt = [mask.numpy().astype(np.uint8) for mask in targets[0]["masks"]]
        for cfg_index, cfg in enumerate(configs):
            pred = postprocess_prepared(prepared, **cfg)
            rows_by_config[cfg_index].append(panoptic_quality(gt, pred))
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results = []
    for cfg, metric_rows in zip(configs, rows_by_config):
        metrics = aggregate(metric_rows)
        results.append({**cfg, **metrics})
    results.sort(key=lambda row: row["pq"], reverse=True)
    for i, row in enumerate(results[:15], 1):
        print(f"{i:02d}. PQ={row['pq']:.5f} score={row['score_threshold']:.2f} mask={row['mask_threshold']:.2f} area={row['min_area']} mode={row['duplicate_mode']} nms={row['mask_nms_iou']:.2f} power={row['overlap_score_power']:.2f} TP={row['tp']} FP={row['fp']} FN={row['fn']} splits={row['splits']} merges={row['merges']}")
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
    parser.add_argument("--out", default="outputs/mask_rcnn/postprocess_sweep_v3.csv")
    args = parser.parse_args()
    run_sweep(args.weights, limit=args.limit, out_csv=args.out)


if __name__ == "__main__":
    main()
