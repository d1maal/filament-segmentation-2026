#!/usr/bin/env python3
"""python run.py check | baseline | train | infer | sweep"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.baseline import predict_instances
from src.config import config
from src.data import (
    ensure_val_split,
    image_stem,
    instances_for_image,
    load_coco,
    mask_to_rle_counts,
    resolve_image,
)
from src.metrics import aggregate, panoptic_quality


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Записывает список словарей в CSV.
    params:
        path: путь к выходному CSV-файлу
        rows: строки для записи
        fieldnames: колонки (по умолчанию filament_id и segmentation_rle)
    returns:
        ничего не возвращает
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or ["filament_id", "segmentation_rle"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _val_items():
    """Формирует список валидационных изображений вместе с COCO-аннотациями и путями.
    returns:
        список кортежей (coco, info, path)
    """
    coco = load_coco()
    val = ensure_val_split(coco)
    items = []
    for info in coco.imgs.values():
        if image_stem(info["file_name"]) not in val:
            continue
        path = resolve_image(config.train_images, info["file_name"])
        if path is None:
            continue
        items.append((coco, info, path))
    items.sort(key=lambda item: image_stem(item[1]["file_name"]))
    return items


def _load_model(weights: Path | None):
    """Загружает чекпоинт и собирает модель с теми же гиперпараметрами, что при train.
    params:
        weights: путь к .pt (по умолчанию last.pt)
    returns:
        weights, checkpoint, processor, model, device
    """
    import torch

    from src.m2f import build

    weights = weights or (config.models_path / "last.pt")
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}. Run training first or pass --weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)
    processor, model = build(
        int(checkpoint.get("size", 512)),
        backbone=checkpoint.get("backbone", "tiny"),
        no_object_weight=float(checkpoint.get("no_object_weight", 0.1)),
        dice_weight=float(checkpoint.get("dice_weight", 5.0)),
        mask_weight=float(checkpoint.get("mask_weight", 5.0)),
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    return weights, checkpoint, processor, model, device


def _predict(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float,
    min_area: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    merge_overlap: float,
):
    """Выбирает полный инференс или тайловый.
    params:
        model: модель Mask2Former
        processor: процессор
        image: grayscale
        device: устройство
        score_thr: порог скора
        min_area: минимальная площадь
        tiled: использовать тайлы
        tile_size: сторона тайла
        tile_overlap: перекрытие
        merge_overlap: порог слияния дублей
    returns:
        список масок
    """
    from src.m2f import predict_masks, predict_masks_tiled

    if tiled:
        return predict_masks_tiled(
            model,
            processor,
            image,
            device,
            score_thr=score_thr,
            min_area=min_area,
            tile_size=tile_size,
            overlap=tile_overlap,
            merge_overlap=merge_overlap,
        )
    return predict_masks(model, processor, image, device, score_thr=score_thr, min_area=min_area)


def _draw_overlay(image: np.ndarray, gt: list[np.ndarray], pred: list[np.ndarray], metrics: dict) -> np.ndarray:
    """Рисует GT (красное), pred (зелёное) и пересечение (жёлтое).
    params:
        image: grayscale
        gt: истинные маски
        pred: предсказанные маски
        metrics: метрики кадра для подписи
    returns:
        BGR-картинка
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
    gt_mask = np.zeros(image.shape[:2], dtype=bool)
    pred_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask in gt:
        gt_mask |= mask > 0
    for mask in pred:
        pred_mask |= mask > 0
    vis[gt_mask & ~pred_mask] = 0.45 * vis[gt_mask & ~pred_mask] + 0.55 * np.array([0, 0, 255])
    vis[pred_mask & ~gt_mask] = 0.45 * vis[pred_mask & ~gt_mask] + 0.55 * np.array([0, 255, 0])
    vis[gt_mask & pred_mask] = 0.45 * vis[gt_mask & pred_mask] + 0.55 * np.array([0, 255, 255])
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    text = (
        f"PQ {metrics['pq']:.3f} | TP {metrics['tp']} | FP {metrics['fp']} | "
        f"FN {metrics['fn']} | split {metrics['splits']} | merge {metrics['merges']}"
    )
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 980), 36), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _summary(rows: list[dict]) -> dict:
    """Агрегирует PQ и добавляет SQ/RQ/precision/recall.
    params:
        rows: покартинные метрики
    returns:
        сводный словарь
    """
    result = aggregate(rows)
    tp = result.get("tp", 0)
    fp = result.get("fp", 0)
    fn = result.get("fn", 0)
    iou_sum = sum(row["sq"] * row["tp"] for row in rows)
    rq_denom = tp + 0.5 * fp + 0.5 * fn
    result["sq"] = float(iou_sum / tp) if tp else 0.0
    result["rq"] = float(tp / rq_denom) if rq_denom else 0.0
    result["precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
    result["recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
    return result


def _append_sweep(results: list[dict], weights: Path, images: int) -> None:
    """Дописывает таблицу sweep в experiments.md.
    params:
        results: строки sweep, уже отсортированные по PQ
        weights: путь к чекпоинту
        images: число картинок val
    returns:
        ничего не возвращает
    """
    path = config.root / "experiments.md"
    lines = [
        "",
        f"## Threshold sweep — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Weights: `{weights}`  ",
        f"Validation images: {images}",
        "",
        "| score_thr | min_area | PQ | SQ | RQ | precision | recall | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['score_thr']:.2f} | {row['min_area']} | {row['pq']:.4f} | "
            f"{row['sq']:.4f} | {row['rq']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Updated {path}")


def cmd_check() -> None:
    """Запускает самопроверку пайплайна: PQ, split, трёхканальность и RLE; данные не требуются."""
    from src.data import mask_to_rle_counts, three_channel

    m: np.ndarray = np.zeros((32, 32), np.uint8)
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
    """Считает Otsu-байзлайн на валидации, опционально рисует viz и пишет submission CSV.
    params:
        submit: флаг генерации сабмишн CSV
        viz: число первых изображений для визуализации (0 — без неё)
    """
    from tqdm import tqdm

    items = _val_items()
    print(f"baseline val: {len(items)} images", flush=True)
    rows = []
    for coco, info, path in tqdm(items, desc="baseline"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Failed to read image: {path}")
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
            out = config.outputs_path / "viz" / f"{metrics['stem']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
    summary = aggregate(rows)
    config.outputs_path.mkdir(parents=True, exist_ok=True)
    (config.outputs_path / "val_metrics.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2), encoding="utf-8"
    )
    print("Home PQ (baseline):", round(summary["pq"], 4), summary)
    if submit:
        csv_rows = []
        for path in sorted(
            p for p in config.test_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SystemExit(f"Failed to read image: {path}")
            for i, mask in enumerate(predict_instances(img), start=1):
                csv_rows.append(
                    {"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)}
                )
        _write_csv(config.submission_baseline_path, csv_rows)


def cmd_train(
    smoke: bool,
    size: int,
    epochs: int,
    max_steps: int,
    accum: int,
    eval_val_samples: int,
    eval_train_samples: int,
    run_name: str,
    augment: bool,
    backbone: str,
    object_crop_prob: float,
    negative_crop_prob: float,
    full_image_epochs: int,
    no_object_weight: float,
    dice_weight: float,
    mask_weight: float,
    warmup_epochs: int,
    lr: float,
    backbone_lr_mult: float,
) -> None:
    """Запускает обучение Mask2Former; в smoke-режиме — 1 эпоха и до 30 шагов.
    params:
        smoke: короткий прогон
        size: сторона кропа
        epochs: число эпох
        max_steps: лимит шагов (0 — без лимита)
        accum: накопление градиента
        eval_val_samples: сколько val-картинок для PQ после эпохи (0 — все)
        eval_train_samples: сколько train-картинок для PQ (0 — не считать)
        run_name: имя run на DagsHub
        augment: геометрические аугментации
        backbone: tiny или small
        object_crop_prob: доля кропов вокруг филамента
        negative_crop_prob: доля пустых кропов
        full_image_epochs: сколько последних эпох на полном диске
        no_object_weight: вес класса «нет объекта»
        dice_weight: вес Dice
        mask_weight: вес mask-лосса
        warmup_epochs: эпохи разогрева lr
        lr: скорость обучения
        backbone_lr_mult: множитель lr энкодера
    """
    from src.m2f import train

    if smoke:
        epochs, max_steps = 1, max_steps or 30
    train(
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        accum=accum,
        eval_val_samples=eval_val_samples,
        eval_train_samples=eval_train_samples,
        run_name=run_name,
        augment=augment,
        backbone=backbone,
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        full_image_epochs=full_image_epochs,
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
        warmup_epochs=warmup_epochs,
        lr=lr,
        backbone_lr_mult=backbone_lr_mult,
    )


def _save_worst(
    rows,
    items,
    model,
    processor,
    device,
    score_thr: float,
    min_area: int,
    count: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    merge_overlap: float,
) -> None:
    """Сохраняет худшие по PQ кадры с overlay.
    params:
        rows: покартинные метрики
        items: те же val-элементы, что в infer
        model: модель
        processor: процессор
        device: устройство
        score_thr: порог скора
        min_area: минимальная площадь
        count: сколько картинок сохранить (0 — ничего)
        tiled: тот же режим, что в infer
        tile_size: сторона тайла
        tile_overlap: перекрытие
        merge_overlap: порог слияния
    """
    if count <= 0:
        return
    by_stem = {image_stem(info["file_name"]): (coco, info, path) for coco, info, path in items}
    worst = sorted(rows, key=lambda row: (row["pq"], -row["fn"], -row["fp"]))[:count]
    mode = "tiled" if tiled else "full"
    out_dir = config.models_path / "viz" / f"{mode}_score_{score_thr:.2f}_area_{min_area}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(worst, start=1):
        coco, info, path = by_stem[row["stem"]]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        pred = _predict(
            model, processor, image, device, score_thr, min_area,
            tiled, tile_size, tile_overlap, merge_overlap,
        )
        vis = _draw_overlay(image, instances_for_image(coco, info["id"]), pred, row)
        cv2.imwrite(str(out_dir / f"{rank:02d}_pq_{row['pq']:.3f}_{row['stem']}.jpg"), vis)
    print(f"Saved {len(worst)} visualizations to {out_dir}")


def cmd_infer(
    submit: bool,
    weights: Path | None,
    score_thr: float,
    min_area: int,
    viz_worst: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    merge_overlap: float,
    limit: int,
) -> None:
    """Прогоняет модель по валидации, считает PQ, опционально пишет submission CSV.
    params:
        submit: генерация сабмишн CSV
        weights: чекпоинт (по умолчанию last.pt)
        score_thr: порог скора
        min_area: минимальная площадь маски
        viz_worst: сколько худших кадров сохранить
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
        merge_overlap: порог слияния дублей
        limit: ограничить val первыми N картинками (0 — все)
    """
    from tqdm import tqdm

    weights, checkpoint, processor, model, device = _load_model(weights)
    items = _val_items()
    if limit > 0:
        items = items[:limit]
    rows = []
    mode = "tiled" if tiled else "full"
    print(
        f"infer val: {len(items)} images | mode={mode} | score={score_thr} | min_area={min_area}",
        flush=True,
    )
    for coco, info, path in tqdm(items, desc="infer"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Failed to read image: {path}")
        pred = _predict(
            model, processor, image, device, score_thr, min_area,
            tiled, tile_size, tile_overlap, merge_overlap,
        )
        metrics = panoptic_quality(instances_for_image(coco, info["id"]), pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    summary = _summary(rows)
    config.models_path.mkdir(parents=True, exist_ok=True)
    name = "val_metrics_m2f_tiled.json" if tiled else "val_metrics_m2f.json"
    (config.models_path / name).write_text(
        json.dumps(
            {
                "config": {
                    "weights": str(weights),
                    "epoch": checkpoint.get("epoch"),
                    "score_thr": score_thr,
                    "min_area": min_area,
                    "tiled": tiled,
                    "images": len(items),
                },
                "summary": summary,
                "per_image": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Home PQ (Mask2Former):", round(summary["pq"], 4), summary)
    _save_worst(
        rows, items, model, processor, device, score_thr, min_area, viz_worst,
        tiled, tile_size, tile_overlap, merge_overlap,
    )
    if submit:
        csv_rows = []
        for path in sorted(
            p for p in config.test_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise SystemExit(f"Failed to read image: {path}")
            pred = _predict(
                model, processor, image, device, score_thr, min_area,
                tiled, tile_size, tile_overlap, merge_overlap,
            )
            for i, mask in enumerate(pred, start=1):
                csv_rows.append(
                    {"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)}
                )
        _write_csv(config.submission_model_path, csv_rows)


def cmd_sweep(
    weights: Path | None, score_thrs: list[float], min_areas: list[int], limit: int, write_log: bool
) -> None:
    """Сетка score_thr × min_area на одном forward на картинку.
    params:
        weights: чекпоинт
        score_thrs: пороги скора
        min_areas: пороги площади
        limit: ограничить val (0 — все)
        write_log: дописать experiments.md
    """
    from tqdm import tqdm

    from src.m2f import predict_masks_grid

    weights, checkpoint, processor, model, device = _load_model(weights)
    items = _val_items()
    if limit > 0:
        items = items[:limit]
    settings = [(float(score), int(area)) for score in score_thrs for area in min_areas]
    rows = {setting: [] for setting in settings}
    print(f"sweep: {len(items)} images, {len(settings)} settings", flush=True)
    for coco, info, path in tqdm(items, desc="sweep"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        gt = instances_for_image(coco, info["id"])
        preds = predict_masks_grid(model, processor, image, device, score_thrs, min_areas)
        stem = image_stem(info["file_name"])
        for setting, pred in preds.items():
            metrics = panoptic_quality(gt, pred)
            metrics["stem"] = stem
            rows[setting].append(metrics)
    results = []
    for score_thr, min_area in settings:
        summary = _summary(rows[(score_thr, min_area)])
        results.append({"score_thr": score_thr, "min_area": min_area, **summary})
    results.sort(key=lambda row: row["pq"], reverse=True)
    config.models_path.mkdir(parents=True, exist_ok=True)
    (config.models_path / "threshold_sweep.json").write_text(
        json.dumps(
            {
                "config": {"weights": str(weights), "epoch": checkpoint.get("epoch"), "images": len(items)},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(
        config.models_path / "threshold_sweep.csv",
        results,
        [
            "score_thr", "min_area", "pq", "sq", "rq", "precision", "recall",
            "tp", "fp", "fn", "splits", "merges", "images",
        ],
    )
    print("\nBest settings:")
    for row in results[:10]:
        print(
            f"score={row['score_thr']:.2f} area={row['min_area']:4d} "
            f"PQ={row['pq']:.4f} SQ={row['sq']:.4f} RQ={row['rq']:.4f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )
    if write_log:
        _append_sweep(results, weights, len(items))


def main() -> None:
    """Точка входа CLI: check, baseline, train, infer, sweep."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
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
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--eval-samples-val", type=int, default=0)
    t.add_argument("--eval-samples-train", type=int, default=0)
    t.add_argument("--run-name", type=str, default="m2f")
    t.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--backbone", choices=["tiny", "small"], default="tiny")
    t.add_argument("--object-crop-prob", type=float, default=0.7)
    t.add_argument("--negative-crop-prob", type=float, default=0.1)
    t.add_argument("--full-image-epochs", type=int, default=3)
    t.add_argument("--no-object-weight", type=float, default=0.08)
    t.add_argument("--dice-weight", type=float, default=5.0)
    t.add_argument("--mask-weight", type=float, default=5.0)
    t.add_argument("--warmup-epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--backbone-lr-mult", type=float, default=0.1)
    i = sub.add_parser("infer")
    i.add_argument("--submit", action="store_true")
    i.add_argument("--weights", type=Path, default=None)
    i.add_argument("--score-thr", type=float, default=0.3)
    i.add_argument("--min-area", type=int, default=80)
    i.add_argument("--viz-worst", type=int, default=0)
    i.add_argument("--tiled", action="store_true")
    i.add_argument("--tile-size", type=int, default=1024)
    i.add_argument("--tile-overlap", type=int, default=256)
    i.add_argument("--merge-overlap", type=float, default=0.6)
    i.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("sweep")
    s.add_argument("--weights", type=Path, default=None)
    s.add_argument("--score-thrs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    s.add_argument("--min-areas", type=int, nargs="+", default=[0, 40, 80, 160])
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-log", action="store_true")
    args = p.parse_args()
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
    main()
