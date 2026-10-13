"""Инференс Swin-модели и команды валидации: infer и sweep."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import config
from src.data import image_stem, instances_for_image, mask_to_rle_counts
from src.eval.submission import _write_csv
from src.eval.val_items import _val_items
from src.io.sweep_log import _append_sweep
from src.metrics import aggregate, panoptic_quality
from src.models.mask_dino_swin.build import OUT_DIR, load_checkpoint
from src.models.mask_dino_swin.dataset import normalize_image

MAX_INSTANCES = 150


def _forward(
    model, device, image: np.ndarray, forward_size: int
) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
    """Один forward: трёхканальный вход, скоры и маски queries в исходном размере.
    params:
        model: обученная модель SwinMaskDINO
        device: устройство выполнения вычислений
        image: grayscale-изображение (или BGR)
        forward_size: сторона квадратного входа сети (кратна 32)
    returns:
        кортеж (маски в размере входа, скоры, исходный размер)
    """
    from src.data import three_channel

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    rgb = three_channel(gray)
    resized = cv2.resize(rgb, (forward_size, forward_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(normalize_image(resized).transpose(2, 0, 1)).unsqueeze(0)
    outputs = model(tensor.to(device))
    scores = outputs["pred_logits"][0].sigmoid().detach().cpu().numpy()
    masks = outputs["pred_masks"][0].sigmoid().detach().cpu().numpy()
    full = []
    for mask in masks:
        full.append(
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        )
    return full, [float(s) for s in scores], (h, w)


@torch.no_grad()
def _predict_scored(
    model,
    device,
    image: np.ndarray,
    score_thr: float,
    min_area: int = 0,
    forward_size: int = 1024,
) -> list[tuple[np.ndarray, float]]:
    """Инференс одного изображения со скорами.
    params:
        model: модель SwinMaskDINO
        device: устройство
        image: grayscale или BGR
        score_thr: минимальный скор
        min_area: минимальная площадь
        forward_size: сторона входа сети
    returns:
        список пар (маска, скор), отсортированный по скору
    """
    masks, scores, _ = _forward(model, device, image, forward_size)
    instances = []
    for mask, score in zip(masks, scores):
        if score < score_thr:
            continue
        binary = (mask >= 0.5).astype(np.uint8)
        if int(binary.sum()) >= min_area:
            instances.append((binary, score))
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances[:MAX_INSTANCES]


@torch.no_grad()
def predict_masks(
    model,
    device,
    image: np.ndarray,
    score_thr: float = 0.3,
    min_area: int = 80,
    forward_size: int = 1024,
) -> list[np.ndarray]:
    """Делает инференс модели на одном полном изображении.
    params:
        model: обученная модель SwinMaskDINO
        device: устройство выполнения вычислений
        image: grayscale-изображение (или BGR)
        score_thr: минимальный скор детекции
        min_area: минимальная площадь маски в пикселях
        forward_size: сторона входа сети
    returns:
        список бинарных масок инстансов
    """
    return [
        mask
        for mask, _ in _predict_scored(model, device, image, score_thr, min_area, forward_size)
    ]


@torch.no_grad()
def predict_masks_grid(
    model,
    device,
    image: np.ndarray,
    score_thrs: list[float],
    min_areas: list[int],
    forward_size: int = 1024,
) -> dict[tuple[float, int], list[np.ndarray]]:
    """Один forward, несколько порогов score/min_area — для sweep.
    params:
        model: модель SwinMaskDINO
        device: устройство
        image: grayscale или BGR
        score_thrs: пороги скора
        min_areas: пороги площади
        forward_size: сторона входа сети
    returns:
        словарь (score_thr, min_area) → маски
    """
    masks, scores, _ = _forward(model, device, image, forward_size)
    areas = [int((mask >= 0.5).sum()) for mask in masks]
    binaries = [(mask >= 0.5).astype(np.uint8) for mask in masks]
    predictions = {}
    for score_thr in score_thrs:
        kept = [
            (binary, area)
            for binary, area, score in zip(binaries, areas, scores)
            if score >= float(score_thr)
        ]
        for min_area in min_areas:
            predictions[(float(score_thr), int(min_area))] = [
                binary for binary, area in kept if area >= int(min_area)
            ][:MAX_INSTANCES]
    return predictions


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


def cmd_infer(
    submit: bool,
    weights: Path | None,
    score_thr: float,
    min_area: int,
    limit: int,
    forward_size: int = 1024,
) -> None:
    """Прогоняет Swin-модель по валидации, считает PQ, опционально пишет submission CSV.
    params:
        submit: генерация сабмишн CSV
        weights: чекпоинт (по умолчанию last.pt)
        score_thr: порог скора
        min_area: минимальная площадь маски
        limit: ограничить val первыми N картинками (0 — все)
        forward_size: сторона входа сети
    """
    from tqdm import tqdm

    weights, checkpoint, model, device = load_checkpoint(weights)
    items = _val_items()
    if limit > 0:
        items = items[:limit]
    print(
        f"infer val: {len(items)} images | score={score_thr} | min_area={min_area}",
        flush=True,
    )
    rows = []
    for coco, info, path in tqdm(items, desc="infer"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Failed to read image: {path}")
        pred = predict_masks(model, device, image, score_thr, min_area, forward_size)
        metrics = panoptic_quality(instances_for_image(coco, info["id"]), pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    summary = _summary(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val_metrics_mask_dino_swin.json").write_text(
        json.dumps(
            {
                "config": {
                    "weights": str(weights),
                    "epoch": checkpoint.get("epoch"),
                    "score_thr": score_thr,
                    "min_area": min_area,
                    "forward_size": forward_size,
                    "images": len(items),
                },
                "summary": summary,
                "per_image": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Home PQ (mask_dino_swin):", round(summary["pq"], 4), summary)
    if submit:
        csv_rows = []
        for path in sorted(
            p for p in config.test_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise SystemExit(f"Failed to read image: {path}")
            pred = predict_masks(model, device, image, score_thr, min_area, forward_size)
            for i, mask in enumerate(pred, start=1):
                csv_rows.append(
                    {"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)}
                )
        _write_csv(config.outputs_path / "submission_mask_dino_swin.csv", csv_rows)


def cmd_sweep(
    weights: Path | None,
    score_thrs: list[float],
    min_areas: list[int],
    limit: int,
    write_log: bool,
    forward_size: int = 1024,
) -> None:
    """Сетка score_thr × min_area на одном forward на картинку.
    params:
        weights: чекпоинт
        score_thrs: пороги скора
        min_areas: пороги площади
        limit: ограничить val (0 — все)
        write_log: дописать experiments.md
        forward_size: сторона входа сети
    """
    from tqdm import tqdm

    weights, checkpoint, model, device = load_checkpoint(weights)
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
        preds = predict_masks_grid(model, device, image, score_thrs, min_areas, forward_size)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "threshold_sweep_mask_dino_swin.json").write_text(
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
        OUT_DIR / "threshold_sweep_mask_dino_swin.csv",
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
