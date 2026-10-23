from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import aggregate, panoptic_quality

from .config import config
from .dataset import FilamentMaskRCNNDataset
from .infer import load_checkpoint, postprocess_prepared, prepare_output
from .utils import collate_fn


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _iou(a, b):
    a = a > 0
    b = b > 0
    box_a = _bbox(a)
    box_b = _bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ay1, ay2, ax1, ax2 = box_a
    by1, by2, bx1, bx2 = box_b
    y1 = max(ay1, by1)
    y2 = min(ay2, by2)
    x1 = max(ax1, bx1)
    x2 = min(ax2, bx2)
    if y1 >= y2 or x1 >= x2:
        return 0.0
    inter = int(np.logical_and(a[y1:y2, x1:x2], b[y1:y2, x1:x2]).sum())
    if inter == 0:
        return 0.0
    area_a = int(a.sum())
    area_b = int(b.sum())
    return float(inter / (area_a + area_b - inter))


def _iou_matrix(a, b):
    matrix = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            matrix[i, j] = _iou(x, y)
    return matrix


def _matches(gt, pred, thr=0.5):
    matrix = _iou_matrix(gt, pred)
    used_g = set()
    used_p = set()
    pairs = []
    if matrix.size:
        for idx in np.argsort(-matrix, axis=None):
            gi, pj = map(int, np.unravel_index(int(idx), matrix.shape))
            value = float(matrix[gi, pj])
            if value < thr:
                break
            if gi in used_g or pj in used_p:
                continue
            used_g.add(gi)
            used_p.add(pj)
            pairs.append((gi, pj, value))
    return matrix, used_g, used_p, pairs


def _raw_masks(prepared, score_floor, mask_threshold, min_area=40):
    masks = []
    scores = []
    for item in prepared:
        if float(item["score"]) < score_floor:
            continue
        mask = (item["prob"] >= mask_threshold).astype(np.uint8)
        if int(mask.sum()) < min_area:
            continue
        masks.append(mask)
        scores.append(float(item["score"]))
    return masks, scores


def _overlay(image, gt, pred, used_g, used_p, text):
    gray = image[0].detach().cpu().numpy()
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, mask in enumerate(gt):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 0, 255) if i in used_g else (255, 0, 255)
        cv2.drawContours(vis, contours, -1, color, 2)
    for i, mask in enumerate(pred):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 255, 0) if i in used_p else (0, 255, 255)
        cv2.drawContours(vis, contours, -1, color, 2)
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1150), 42), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _recommend(summary):
    notes = []
    fn = int(summary["fn"])
    fp = int(summary["fp"])
    recoverable = int(summary["recoverable_fn"])
    hallucination = int(summary["hallucination_fp"])
    delta = float(summary["disjoint_pq_delta"])
    if fn and recoverable / fn >= 0.30:
        notes.append("A large part of FN already has a raw candidate with IoU >= 0.5; selector/postprocess is too strict for those cases.")
    if fp and hallucination / fp >= 0.40:
        notes.append("A large part of FP has IoU < 0.1 with every GT object; stronger selector confidence/TTA agreement is likely useful.")
    if delta < -0.01:
        notes.append("Disjoint arbitration noticeably reduces PQ; overlap ownership parameters need tuning.")
    if int(summary["unrecoverable_fn"]) > int(summary["recoverable_fn"]):
        notes.append("Most FN have no raw candidate at IoU >= 0.5; the next gain must come from the model/training rather than selector only.")
    if not notes:
        notes.append("No single error family dominates; tune selector on full validation and inspect the worst overlays before changing the model.")
    return notes


@torch.no_grad()
def run_autopsy(weights, limit=0, overlays=20, out_dir=None):
    out_dir = Path(out_dir or config.output_dir / "autopsy_v5")
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_checkpoint(weights)
    dataset = FilamentMaskRCNNDataset(train=False, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    metric_rows = []
    pre_disjoint_rows = []
    per_image = []
    overlay_items = []
    recoverable_fn_total = 0
    unrecoverable_fn_total = 0
    hallucination_fp_total = 0
    near_fp_total = 0
    raw_oracle_gt_total = 0
    gt_total = 0

    for image_index, (images, targets) in enumerate(tqdm(loader, desc="autopsy")):
        if limit and image_index >= limit:
            break

        image = images[0]
        output = model([image.to(device)])[0]
        prepared = prepare_output(output, score_floor=config.autopsy_score_floor)
        gt = [mask.numpy().astype(np.uint8) for mask in targets[0]["masks"]]

        pred = postprocess_prepared(prepared)
        pred_pre = postprocess_prepared(prepared, enforce_disjoint=False)
        metrics = panoptic_quality(gt, pred)
        pre_metrics = panoptic_quality(gt, pred_pre)
        metric_rows.append(metrics)
        pre_disjoint_rows.append(pre_metrics)

        matrix, used_g, used_p, _ = _matches(gt, pred)
        raw_masks, _ = _raw_masks(
            prepared,
            score_floor=config.autopsy_score_floor,
            mask_threshold=config.mask_threshold,
            min_area=40,
        )
        raw_matrix = _iou_matrix(gt, raw_masks)

        recoverable_fn = 0
        unrecoverable_fn = 0
        for gi in range(len(gt)):
            raw_best = float(raw_matrix[gi].max()) if raw_matrix.shape[1] else 0.0
            if raw_best >= 0.5:
                raw_oracle_gt_total += 1
            if gi not in used_g:
                if raw_best >= 0.5:
                    recoverable_fn += 1
                else:
                    unrecoverable_fn += 1

        hallucination_fp = 0
        near_fp = 0
        for pj in range(len(pred)):
            if pj in used_p:
                continue
            best_iou = float(matrix[:, pj].max()) if matrix.shape[0] else 0.0
            if best_iou < 0.10:
                hallucination_fp += 1
            else:
                near_fp += 1

        recoverable_fn_total += recoverable_fn
        unrecoverable_fn_total += unrecoverable_fn
        hallucination_fp_total += hallucination_fp
        near_fp_total += near_fp
        gt_total += len(gt)

        stem = Path(dataset.rows[image_index][0]["file_name"]).stem
        row = {
            "stem": stem,
            **metrics,
            "pre_disjoint_pq": float(pre_metrics["pq"]),
            "pq_delta_disjoint": float(metrics["pq"] - pre_metrics["pq"]),
            "recoverable_fn": recoverable_fn,
            "unrecoverable_fn": unrecoverable_fn,
            "hallucination_fp": hallucination_fp,
            "near_fp": near_fp,
            "raw_candidates": len(raw_masks),
        }
        per_image.append(row)
        overlay_items.append((float(metrics["pq"]), stem, image, gt, pred, used_g, used_p, row))

    summary = aggregate(metric_rows)
    pre_summary = aggregate(pre_disjoint_rows)
    summary.update(
        {
            "pre_disjoint_pq": float(pre_summary.get("pq", 0.0)),
            "disjoint_pq_delta": float(summary.get("pq", 0.0) - pre_summary.get("pq", 0.0)),
            "recoverable_fn": int(recoverable_fn_total),
            "unrecoverable_fn": int(unrecoverable_fn_total),
            "hallucination_fp": int(hallucination_fp_total),
            "near_fp": int(near_fp_total),
            "raw_oracle_recall_at_05": float(raw_oracle_gt_total / max(1, gt_total)),
            "gt_total": int(gt_total),
        }
    )

    overlay_items.sort(key=lambda item: (item[0], -item[7]["fn"], -item[7]["fp"]))
    saved = []
    for rank, item in enumerate(overlay_items[: int(overlays)], 1):
        pq, stem, image, gt, pred, used_g, used_p, row = item
        text = (
            f"PQ={pq:.3f} TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"recFN={row['recoverable_fn']} hardFN={row['unrecoverable_fn']}"
        )
        vis = _overlay(image, gt, pred, used_g, used_p, text)
        path = overlay_dir / f"{rank:02d}_pq_{pq:.3f}_{stem}.jpg"
        cv2.imwrite(str(path), vis)
        saved.append(path.name)

    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()) if per_image else ["stem"])
        writer.writeheader()
        writer.writerows(per_image)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    recommendations = _recommend(summary)
    report = [
        "# Mask R-CNN v5 error autopsy",
        "",
        f"PQ: {summary.get('pq', 0.0):.5f}",
        f"TP / FP / FN: {summary.get('tp', 0)} / {summary.get('fp', 0)} / {summary.get('fn', 0)}",
        f"Splits / merges: {summary.get('splits', 0)} / {summary.get('merges', 0)}",
        f"Pre-disjoint PQ: {summary.get('pre_disjoint_pq', 0.0):.5f}",
        f"Disjoint PQ delta: {summary.get('disjoint_pq_delta', 0.0):+.5f}",
        f"Recoverable FN: {summary.get('recoverable_fn', 0)}",
        f"Unrecoverable FN: {summary.get('unrecoverable_fn', 0)}",
        f"Hallucination FP: {summary.get('hallucination_fp', 0)}",
        f"Near-match FP: {summary.get('near_fp', 0)}",
        f"Raw candidate oracle recall@0.5: {summary.get('raw_oracle_recall_at_05', 0.0):.4f}",
        "",
        "## Conclusions",
        "",
    ]
    report.extend(f"- {note}" for note in recommendations)
    report.extend(["", "## Worst overlays", ""])
    report.extend(f"- overlays/{name}" for name in saved)
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"report: {out_dir / 'report.md'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overlays", type=int, default=config.autopsy_overlays)
    parser.add_argument("--out", default=str(config.output_dir / "autopsy_v5"))
    args = parser.parse_args()
    run_autopsy(args.weights, limit=args.limit, overlays=args.overlays, out_dir=args.out)


if __name__ == "__main__":
    main()
