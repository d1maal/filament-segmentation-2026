from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.config import config as project_config
from src.data import mask_to_rle_counts, three_channel

from .config import config
from .infer import load_checkpoint, predict_masks


def write_submission(weights, output=None, tta="full", **predict_kwargs):
    output = Path(output or config.output_dir / "submission_mask_rcnn_v5.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    model, device = load_checkpoint(weights)
    rows = []
    paths = sorted(
        p
        for p in project_config.test_images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    for path in tqdm(paths, desc="submit"):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image = three_channel(gray)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        masks = predict_masks(model, tensor, device, tta=tta, **predict_kwargs)
        for i, mask in enumerate(masks, 1):
            rows.append(
                {
                    "filament_id": f"{path.stem}_{i}",
                    "segmentation_rle": mask_to_rle_counts(mask),
                }
            )
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output} rows={len(rows)}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default=str(config.output_dir / "submission_mask_rcnn_v5.csv"))
    parser.add_argument("--tta", choices=["none", "light", "full"], default="full")
    parser.add_argument("--score", type=float, default=config.score_threshold)
    parser.add_argument("--mask", type=float, default=config.mask_threshold)
    parser.add_argument("--area", type=int, default=config.min_area)
    parser.add_argument("--votes", type=int, default=config.min_tta_votes)
    parser.add_argument("--singleton", type=float, default=config.single_view_score_threshold)
    parser.add_argument("--retained", type=float, default=config.min_retained_fraction)
    parser.add_argument("--overlap-weight", type=float, default=config.overlap_score_weight)
    args = parser.parse_args()
    tta = False if args.tta == "none" else args.tta
    write_submission(
        args.weights,
        output=args.output,
        tta=tta,
        score_threshold=args.score,
        mask_threshold=args.mask,
        min_area=args.area,
        min_tta_votes=args.votes,
        single_view_score_threshold=args.singleton,
        min_retained_fraction=args.retained,
        overlap_score_weight=args.overlap_weight,
    )


if __name__ == "__main__":
    main()
