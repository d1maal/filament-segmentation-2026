#!/usr/bin/env python3
"""python run.py check | baseline | train | infer"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.baseline import predict_instances
from src.data import (
    OUTPUTS,
    TEST_IMAGES,
    TRAIN_ANN,
    TRAIN_IMAGES,
    ensure_val_split,
    image_stem,
    instances_for_image,
    load_coco,
    mask_to_rle_counts,
    resolve_image,
)
from src.metrics import aggregate, panoptic_quality


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["filament_id", "segmentation_rle"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _val_items():
    coco = load_coco()
    val = ensure_val_split(coco)
    items = []
    for info in coco.imgs.values():
        if image_stem(info["file_name"]) not in val:
            continue
        path = resolve_image(TRAIN_IMAGES, info["file_name"])
        if path is None:
            continue
        items.append((coco, info, path))
    return items


def cmd_check() -> None:
    from src.data import mask_to_rle_counts, three_channel

    m = np.zeros((32, 32), np.uint8)
    m[5:15, 5:20] = 1
    assert panoptic_quality([m], [m.copy()])["pq"] == 1.0
    a, b = np.zeros_like(m), np.zeros_like(m)
    a[5:15, 5:12] = 1
    b[5:15, 14:20] = 1
    assert panoptic_quality([m], [a, b])["splits"] == 1
    rgb = three_channel(np.full((64, 64), 120, np.uint8))
    assert rgb.shape == (64, 64, 3)
    assert mask_to_rle_counts(m)
    print("check ok")


def cmd_baseline(submit: bool, viz: int) -> None:
    from tqdm import tqdm

    items = _val_items()
    print(f"baseline val: {len(items)} images", flush=True)
    rows = []
    for coco, info, path in tqdm(items, desc="baseline"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = instances_for_image(coco, info["id"])
        pred = predict_instances(img)
        metrics = panoptic_quality(gt, pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
        if viz and len(rows) <= viz:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for g in gt:
                vis[g > 0] = (0.45 * vis[g > 0] + (0, 0, 180)).astype(np.uint8)
            for p in pred:
                vis[p > 0] = (0.45 * vis[p > 0] + (0, 180, 0)).astype(np.uint8)
            out = OUTPUTS / "viz" / f"{metrics['stem']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
    summary = aggregate(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "val_metrics.json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2), encoding="utf-8")
    print("Home PQ (baseline):", round(summary["pq"], 4), summary)
    if submit:
        csv_rows = []
        for path in sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            for i, mask in enumerate(predict_instances(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)), start=1):
                csv_rows.append({"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)})
        _write_csv(OUTPUTS / "submission_baseline.csv", csv_rows)


def cmd_train(smoke: bool, size: int, epochs: int, max_steps: int) -> None:
    from src.m2f import train

    if smoke:
        epochs, max_steps = 1, max_steps or 30
    train(size=size, epochs=epochs, max_steps=max_steps)


def cmd_infer(submit: bool, weights: Path | None) -> None:
    import torch
    from tqdm import tqdm
    from src.m2f import build, predict_masks

    weights = weights or (OUTPUTS / "m2f" / "last.pt")
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}. Run: python run.py train --smoke")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        bundle = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(weights, map_location=device)
    processor, model = build(int(bundle.get("size", 512)))
    model.load_state_dict(bundle["model"], strict=False)
    model.to(device).eval()

    rows = []
    items = _val_items()
    print(f"infer val: {len(items)} images", flush=True)
    for coco, info, path in tqdm(items, desc="infer"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = instances_for_image(coco, info["id"])
        pred = predict_masks(model, processor, img, device)
        metrics = panoptic_quality(gt, pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    summary = aggregate(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "val_metrics_m2f.json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2), encoding="utf-8")
    print("Home PQ (Mask2Former):", round(summary["pq"], 4), summary)
    if submit:
        csv_rows = []
        for path in sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            pred = predict_masks(model, processor, cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), device)
            for i, mask in enumerate(pred, start=1):
                csv_rows.append({"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)})
        _write_csv(OUTPUTS / "submission_m2f.csv", csv_rows)


def main() -> None:
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
    t.add_argument("--max-steps", type=int, default=0)
    i = sub.add_parser("infer")
    i.add_argument("--submit", action="store_true")
    i.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "baseline":
        cmd_baseline(args.submit, args.viz)
    elif args.cmd == "train":
        cmd_train(args.smoke, args.size, args.epochs, args.max_steps)
    else:
        cmd_infer(args.submit, args.weights)


if __name__ == "__main__":
    main()
