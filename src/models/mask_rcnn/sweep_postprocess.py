from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import aggregate, panoptic_quality

from .dataset import FilamentMaskRCNNDataset
from .infer import load_checkpoint, suppress_duplicate_masks
from .utils import collate_fn


def _configs():
    score_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    min_areas = [80, 150, 250]

    configs = []
    for score_thr, min_area in product(score_thresholds, min_areas):
        configs.append(
            {
                "score_threshold": score_thr,
                "min_area": min_area,
                "deduplicate": False,
                "mask_nms_iou": 0.60,
                "containment_threshold": 0.90,
            }
        )

    # Conservative duplicate suppression variants.
    for score_thr, min_area in [
        (0.35, 80),
        (0.40, 80),
        (0.40, 150),
        (0.45, 80),
        (0.45, 150),
        (0.50, 150),
    ]:
        for nms_iou in [0.60, 0.70]:
            configs.append(
                {
                    "score_threshold": score_thr,
                    "min_area": min_area,
                    "deduplicate": True,
                    "mask_nms_iou": nms_iou,
                    "containment_threshold": 0.90,
                }
            )
    return configs


def _prepare_candidates(output, score_floor=0.25, mask_threshold=0.50):
    """
    Move/binarize candidate masks once per image.
    The sweep then reuses these masks for every threshold combination.
    """
    scores_tensor = output["scores"].detach()
    keep = scores_tensor >= float(score_floor)

    if int(keep.sum()) == 0:
        return [], [], []

    scores = scores_tensor[keep].cpu().numpy().astype(float).tolist()
    masks_tensor = output["masks"][keep, 0].detach().cpu()
    masks = [
        (mask.numpy() >= mask_threshold).astype(np.uint8)
        for mask in masks_tensor
    ]
    areas = [int(mask.sum()) for mask in masks]
    return masks, scores, areas


def _select(prepared, cfg):
    masks, scores, areas = prepared
    chosen_masks = []
    chosen_scores = []

    for mask, score, area in zip(masks, scores, areas):
        if score < cfg["score_threshold"]:
            continue
        if area < cfg["min_area"]:
            continue
        chosen_masks.append(mask)
        chosen_scores.append(score)

    if cfg["deduplicate"]:
        chosen_masks, chosen_scores = suppress_duplicate_masks(
            chosen_masks,
            chosen_scores,
            iou_threshold=cfg["mask_nms_iou"],
            containment_threshold=cfg["containment_threshold"],
        )

    return chosen_masks


@torch.no_grad()
def run_sweep(weights: str | Path, limit: int = 0, out_csv: str | Path | None = None):
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
    rows_by_config = [[] for _ in configs]

    for image_index, (images, targets) in enumerate(tqdm(loader, desc="sweep val")):
        if limit and image_index >= limit:
            break

        image = images[0].to(device)
        output = model([image])[0]
        prepared = _prepare_candidates(output)
        gt = [m.numpy().astype(np.uint8) for m in targets[0]["masks"]]

        for cfg_index, cfg in enumerate(configs):
            pred = _select(prepared, cfg)
            rows_by_config[cfg_index].append(panoptic_quality(gt, pred))

        del output, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = []
    for cfg, metric_rows in zip(configs, rows_by_config):
        metrics = aggregate(metric_rows)
        results.append({**cfg, **metrics})

    results.sort(key=lambda row: row["pq"], reverse=True)

    print("\nTOP CONFIGS")
    for i, row in enumerate(results[:12], 1):
        print(
            f"{i:02d}. PQ={row['pq']:.4f} "
            f"score={row['score_threshold']:.2f} "
            f"area={row['min_area']} "
            f"dedup={row['deduplicate']} "
            f"nms={row['mask_nms_iou']:.2f} "
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
        print(f"\nSaved: {out_csv}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        default="outputs/mask_rcnn/postprocess_sweep.csv",
    )
    args = parser.parse_args()

    run_sweep(args.weights, limit=args.limit, out_csv=args.out)


if __name__ == "__main__":
    main()
