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


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Записывает список строк в CSV-файл с колонками filament_id и segmentation_rle и сообщает путь и число строк.
    params:
        path: путь к выходному CSV-файлу
        rows: список словарей с данными для записи
    returns:
        ничего не возвращает
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["filament_id", "segmentation_rle"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _val_items():
    """Формирует список валидационных изображений вместе с их COCO-аннотациями и разрешёнными путями к файлам.
    returns:
        список кортежей (coco, info, path) для валидационных изображений
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
    return items


def cmd_check() -> None:
    """Запускает самопроверку пайплайна: корректность PQ, детекции split, трёхканальности и RLE-кодирования; данные не требуются."""
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
    """Считает и сохраняет метрики Otsu-байзлайна на валидации, опционально рисует визуализации и генерирует сабмишн CSV по тестовым изображениям.
    params:
        submit: флаг генерации сабмишн CSV
        viz: число первых изображений для сохранения визуализации (0 — без неё)
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
            p
            for p in config.test_images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SystemExit(f"Failed to read image: {path}")
            for i, mask in enumerate(predict_instances(img), start=1):
                csv_rows.append(
                    {
                        "filament_id": f"{path.stem}_{i}",
                        "segmentation_rle": mask_to_rle_counts(mask),
                    }
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
    run_name: str = "m2f",
) -> None:
    """Запускает обучение Mask2Former с заданными параметрами; в smoke-режиме ограничивает обучение одной эпохой и тридцатью шагами.
    params:
        smoke: флаг быстрого проверочного прогона
        size: сторона кропа при обучении
        epochs: число эпох
        max_steps: ограничение по числу шагов
        accum: число шагов накопления градиента
        eval_val_samples: число случайных изображений для val PQ (0 — вся валидация)
        eval_train_samples: число случайных изображений для train PQ (0 — не считать)
        run_name: имя MLflow run на DagsHub (по умолчанию 'm2f')
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
    )


def cmd_infer(submit: bool, weights: Path | None) -> None:
    """Прогоняет обученную модель по валидации, считает и сохраняет PQ, опционально генерирует сабмишн CSV.
    params:
        submit: флаг генерации сабмишн CSV
        weights: путь к чекпоинту модели (по умолчанию last.pt в папке моделей)
    """
    import torch
    from tqdm import tqdm

    from src.m2f import build, predict_masks

    weights = weights or (config.models_path / "last.pt")
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
        if img is None:
            raise SystemExit(f"Failed to read image: {path}")
        gt = instances_for_image(coco, info["id"])
        pred = predict_masks(model, processor, img, device)
        metrics = panoptic_quality(gt, pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    summary = aggregate(rows)
    config.outputs_path.mkdir(parents=True, exist_ok=True)
    (config.outputs_path / "val_metrics_m2f.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2), encoding="utf-8"
    )
    print("Home PQ (Mask2Former):", round(summary["pq"], 4), summary)
    if submit:
        csv_rows = []
        for path in sorted(
            p
            for p in config.test_images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SystemExit(f"Failed to read image: {path}")
            pred = predict_masks(model, processor, img, device)
            for i, mask in enumerate(pred, start=1):
                csv_rows.append(
                    {
                        "filament_id": f"{path.stem}_{i}",
                        "segmentation_rle": mask_to_rle_counts(mask),
                    }
                )
        _write_csv(config.submission_model_path, csv_rows)


def main() -> None:
    """Точка входа CLI: разбирает аргументы и диспетчеризует команды check, baseline, train и infer."""
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
    t.add_argument("--accum", type=int, default=8, help="Gradient accumulation steps")
    t.add_argument(
        "--eval-samples-val",
        type=int,
        default=0,
        help="Random val images for PQ (0 = all)",
    )
    t.add_argument(
        "--eval-samples-train",
        type=int,
        default=0,
        help="Random train images for PQ (0 = skip train metrics)",
    )
    t.add_argument(
        "--run-name",
        type=str,
        default="m2f"        
    )
    i = sub.add_parser("infer")
    i.add_argument("--submit", action="store_true")
    i.add_argument("--weights", type=Path, default=None)
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
        )
    elif args.cmd == "infer":
        cmd_infer(args.submit, args.weights)
    else:
        p.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
