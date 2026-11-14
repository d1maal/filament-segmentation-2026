"""Validation statistics and figures for the technical report.

Runs the released checkpoint once per validation image with all 8 TTA views and
scores three inference settings from the same forward passes:
  none  - identity view only
  light - identity, hflip, vflip, rot180
  full  - all 8 views (setting used for the final submission)

Outputs (default: outputs/report/):
  per_image_<mode>.csv   PQ, TP, FP, FN, splits, merges per validation image
  matches_full.csv       IoU and Dice of every matched (TP) instance, full TTA
  summary.json           aggregated metrics per mode
  overlays/*.png         ground truth (green) vs prediction (red) for selected images

    python scripts/evaluate_val.py --limit 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "weights"))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from assemble import assemble
from src.config import config as project_config
from src.metrics import aggregate, panoptic_quality
from src.models.mask_rcnn.config import config as mcfg
from src.models.mask_rcnn.dataset import FilamentMaskRCNNDataset
from src.models.mask_rcnn.infer import load_checkpoint, postprocess_prepared, prepare_tta

MODES = {
    "none": {"identity"},
    "light": {"identity", "hflip", "vflip", "rot180"},
    "full": set(mcfg.tta_transforms_full),
}


def match_ious(gt, pred, thr=0.5):
    """Greedy one-to-one matching by IoU (same rule as src/metrics.py); returns (iou, dice) of matches."""
    gt = [g > 0 for g in gt]
    pred = [p > 0 for p in pred]
    if not gt or not pred:
        return []
    ga = [int(g.sum()) for g in gt]
    pa = [int(p.sum()) for p in pred]
    iou = np.zeros((len(gt), len(pred)))
    inter = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            n = int(np.logical_and(g, p).sum())
            inter[i, j] = n
            union = ga[i] + pa[j] - n
            iou[i, j] = n / union if union else 0.0
    used_g, used_p, out = set(), set(), []
    for idx in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(int(idx), iou.shape)
        if iou[i, j] < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        dice = 2 * inter[i, j] / (ga[i] + pa[j])
        out.append((float(iou[i, j]), float(dice), ga[i]))
    return out


def overlay(image, gt, pred, title, path, crop=True):
    gray = (image[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for m in gt:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (60, 200, 60), 5)
    for m in pred:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (40, 40, 230), 3)
    if crop:
        # keep the solar disk: crop to the bounding box of non-dark pixels
        ys, xs = np.nonzero(gray > 20)
        if xs.size:
            vis = vis[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    vis = cv2.resize(vis, (1024, int(1024 * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), vis)
    return title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all validation images")
    ap.add_argument("--out", default=str(Path(project_config.outputs_path) / "report"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    weights = assemble()
    model, device = load_checkpoint(weights)
    ds = FilamentMaskRCNNDataset(train=False, augment=False)
    n = len(ds) if not args.limit else min(args.limit, len(ds))

    rows = {m: [] for m in MODES}
    matches = []
    keep = []  # (pq_full, index) to pick overlay examples
    t_infer = []
    for i in tqdm(range(n), desc="val"):
        image, target = ds[i]
        stem = Path(ds.rows[i][0]["file_name"]).stem
        gt = [m.numpy().astype(np.uint8) for m in target["masks"]]
        t0 = time.time()
        prepared = prepare_tta(model, image, device, transforms=mcfg.tta_transforms_full,
                               score_floor=mcfg.score_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer.append(time.time() - t0)
        for mode, views in MODES.items():
            sub = [p for p in prepared if p["sources"] <= views]
            pred = postprocess_prepared(sub)
            r = panoptic_quality(gt, pred)
            r.update({"stem": stem, "n_gt": len(gt), "n_pred": len(pred)})
            rows[mode].append(r)
            if mode == "full":
                for iou, dice, area in match_ious(gt, pred):
                    matches.append({"stem": stem, "iou": iou, "dice": dice, "gt_area": area})
                keep.append((r["pq"], i))

    summary = {}
    for mode, rs in rows.items():
        s = aggregate(rs)
        s["images"] = len(rs)
        summary[mode] = s
        keys = ["stem", "pq", "tp", "fp", "fn", "splits", "merges", "n_gt", "n_pred"]
        with open(out / f"per_image_{mode}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rs)
    summary["seconds_per_image_8_views"] = float(np.mean(t_infer)) if t_infer else None
    summary["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    with open(out / "matches_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "iou", "dice", "gt_area"])
        w.writeheader()
        w.writerows(matches)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # overlays: best, two around the median, worst (images with at least 2 GT filaments)
    cand = sorted((pq, i) for pq, i in keep if len(ds[i][1]["masks"]) >= 2)
    if cand:
        picks = {"best": cand[-1], "median": cand[len(cand) // 2],
                 "q25": cand[len(cand) // 4], "worst": cand[0]}
        for name, (pq, i) in picks.items():
            image, target = ds[i]
            stem = Path(ds.rows[i][0]["file_name"]).stem
            prepared = prepare_tta(model, image, device, transforms=mcfg.tta_transforms_full,
                                   score_floor=mcfg.score_threshold)
            pred = postprocess_prepared(prepared)
            gt = [m.numpy().astype(np.uint8) for m in target["masks"]]
            overlay(image, gt, pred, stem, out / "overlays" / f"{name}_pq{pq:.2f}_{stem}.png")
        print("overlays:", sorted(p.name for p in (out / "overlays").iterdir()))


if __name__ == "__main__":
    main()
