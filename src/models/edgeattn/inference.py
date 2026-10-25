from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import config
from src.data import image_stem, instances_for_image, mask_to_rle_counts
from src.eval.submission import _write_csv
from src.io.overlay import _draw_overlay
from src.metrics import aggregate
from src.models.edgeattn.build import build_arch
from src.models.edgeattn.dataset import make_input, unique_val_items
from src.models.edgeattn.pq_fast import (
    instances_from_labels,
    panoptic_quality_from_labels,
)
from src.models.m2f.tiling import _tile_starts

DEPTH = 16


def mask_to_labels(prob: np.ndarray, thr: float = 0.5, min_area: int = 0) -> np.ndarray:
    """Превращает карту вероятностей в карту меток инстансов.
    params:
        prob: вероятность филамента (H, W) в [0, 1]
        thr: порог бинаризации
        min_area: минимальная площадь компоненты
    returns:
        int32-карта (H, W): 0 — фон, 1..N — инстансы
    """
    binary = (prob >= thr).astype(np.uint8)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = [k for k in range(1, n) if int(stats[k, cv2.CC_STAT_AREA]) >= min_area]
    if not keep:
        return np.zeros(binary.shape, dtype=np.int32)
    mapping: np.ndarray = np.zeros(n, dtype=np.int32)
    for new, old in enumerate(keep, start=1):
        mapping[old] = new
    return mapping[labels].astype(np.int32)


def _check_tiling(tile_size: int, tile_overlap: int) -> None:
    """Проверяет, что параметры тайлинга согласованы с сетью.
    params:
        tile_size: сторона тайла
        tile_overlap: перекрытие тайлов
    """
    if tile_size % DEPTH:
        raise ValueError(f"tile_size={tile_size} must be divisible by {DEPTH}")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap={tile_overlap} must be in [0, {tile_size})")
    if tile_overlap % DEPTH:
        raise ValueError(f"tile_overlap={tile_overlap} must be divisible by {DEPTH}")


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    size: int,
    in_ch: int,
    tiled: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 128,
) -> np.ndarray:
    """Считает карту вероятностей филаментов для целого диска.
    params:
        model: модель EdgeAttNet
        gray: grayscale-изображение (H, W)
        device: устройство
        size: сторона входа при инференсе без тайлов
        in_ch: число каналов модели
        tiled: тайловый инференс с усреднением вероятностей
        tile_size: сторона тайла
        tile_overlap: перекрытие тайлов
    returns:
        float32-карта вероятностей (H, W)
    """
    model.eval()
    h, w = gray.shape
    use_amp = device.type == "cuda"
    if not tiled:
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        x = make_input(resized, in_ch)
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(
            0
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(tensor.to(device))
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    _check_tiling(tile_size, tile_overlap)
    pad_h = (-h) % DEPTH
    pad_w = (-w) % DEPTH
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape
    accum = np.zeros((ph, pw), dtype=np.float32)
    counts = np.zeros((ph, pw), dtype=np.float32)
    for y in _tile_starts(ph, tile_size, tile_overlap):
        for x in _tile_starts(pw, tile_size, tile_overlap):
            tile = padded[y : y + tile_size, x : x + tile_size]
            x_in = make_input(tile, in_ch)
            tensor = torch.from_numpy(
                np.ascontiguousarray(x_in.transpose(2, 0, 1))
            ).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(tensor.to(device))
            prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            accum[y : y + prob.shape[0], x : x + prob.shape[1]] += prob
            counts[y : y + prob.shape[0], x : x + prob.shape[1]] += 1.0
    return (accum / np.maximum(counts, 1.0))[:h, :w]


def load_checkpoint(
    weights: Path | None = None,
) -> tuple[Path, dict, torch.nn.Module, torch.device]:
    """Загружает чекпоинт EdgeAttNet и собирает модель.
    params:
        weights: путь к .pt (по умолчанию best.pt run'а edgeattn)
    returns:
        weights, checkpoint, model, device
    """
    path = Path(weights) if weights else config.edgeattn_default_weights
    if not path.exists():
        raise SystemExit(f"No weights at {path}. Run training first or pass --weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = build_arch(
        arch=checkpoint.get("arch", "edgeattn"),
        in_ch=int(checkpoint.get("in_ch", 3)),
        base=int(checkpoint.get("base", 64)),
        blocks=int(checkpoint.get("blocks", 2)),
        heads=int(checkpoint.get("heads", 4)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return path, checkpoint, model, device


def _predict_labels(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    checkpoint: dict,
    score_thr: float,
    min_area: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    """Считает вероятности и превращает их в карту меток.
    params:
        model: модель
        gray: grayscale-изображение
        device: устройство
        checkpoint: чекпоинт (нужен size и in_ch)
        score_thr: порог вероятности
        min_area: минимальная площадь инстанса
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
    returns:
        карта меток int32
    """
    prob = predict_probability(
        model,
        gray,
        device,
        size=int(checkpoint.get("size", 512)),
        in_ch=int(checkpoint.get("in_ch", 3)),
        tiled=tiled,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    return mask_to_labels(prob, score_thr, min_area)


def _eval_rows(
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    items: list[tuple],
    score_thr: float,
    min_area: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    desc: str,
    leave: bool = True,
) -> list[dict]:
    """Считает покартинные метрики на наборе val/train-изображений.
    params:
        model: модель
        checkpoint: чекпоинт
        device: устройство
        items: список (coco, info, path)
        score_thr: порог вероятности
        min_area: минимальная площадь инстанса
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
        desc: подпись прогресс-бара
        leave: оставлять ли бар после завершения
    returns:
        список словарей метрик с добавленным stem
    """
    rows = []
    for coco, info, path in tqdm(items, desc=desc, leave=leave):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"Failed to read image: {path}")
        labels = _predict_labels(
            model,
            gray,
            device,
            checkpoint,
            score_thr,
            min_area,
            tiled,
            tile_size,
            tile_overlap,
        )
        metrics = panoptic_quality_from_labels(
            instances_for_image(coco, info["id"]), labels
        )
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    return rows


def _summary(rows: list[dict]) -> dict:
    """Агрегирует PQ и добавляет SQ/RQ/precision/recall.
    params:
        rows: покартинные метрики
    returns:
        сводный словарь
    """
    result = aggregate(rows)
    tp = int(result.get("tp", 0))
    fp = int(result.get("fp", 0))
    fn = int(result.get("fn", 0))
    iou_sum = sum(row["sq"] * row["tp"] for row in rows)
    rq_denom = tp + 0.5 * fp + 0.5 * fn
    result["sq"] = float(iou_sum / tp) if tp else 0.0
    result["rq"] = float(tp / rq_denom) if rq_denom else 0.0
    result["precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
    result["recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
    return result


def _save_worst(
    rows: list[dict],
    items: list[tuple],
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    score_thr: float,
    min_area: int,
    count: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
) -> None:
    """Сохраняет худшие по PQ кадры с overlay GT/pred.
    params:
        rows: покартинные метрики
        items: те же элементы, что в infer
        model: модель
        checkpoint: чекпоинт
        device: устройство
        score_thr: порог вероятности
        min_area: минимальная площадь
        count: сколько кадров сохранить (0 — ничего)
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
    """
    if count <= 0:
        return
    by_stem = {
        image_stem(info["file_name"]): (coco, info, path) for coco, info, path in items
    }
    worst = sorted(rows, key=lambda row: (row["pq"], -row["fn"], -row["fp"]))[:count]
    mode = "tiled" if tiled else "full"
    out_dir = (
        config.edgeattn_path / "viz" / f"{mode}_score_{score_thr:.2f}_area_{min_area}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(worst, start=1):
        coco, info, path = by_stem[row["stem"]]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        labels = _predict_labels(
            model,
            gray,
            device,
            checkpoint,
            score_thr,
            min_area,
            tiled,
            tile_size,
            tile_overlap,
        )
        vis = _draw_overlay(
            gray,
            instances_for_image(coco, info["id"]),
            instances_from_labels(labels),
            row,
        )
        cv2.imwrite(
            str(out_dir / f"{rank:02d}_pq_{row['pq']:.3f}_{row['stem']}.jpg"), vis
        )
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
    limit: int,
) -> None:
    """Считает инстансы по val, пишет метрики и опционально submission CSV.
    params:
        submit: генерация submission CSV по test-изображениям
        weights: чекпоинт (по умолчанию best.pt run'а edgeattn)
        score_thr: порог вероятности
        min_area: минимальная площадь инстанса
        viz_worst: сколько худших кадров сохранить
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие тайлов
        limit: ограничить число val- и test-изображений (0 — все)
    """
    weights, checkpoint, model, device = load_checkpoint(weights)
    items = unique_val_items()
    if limit > 0:
        items = items[:limit]
    mode = "tiled" if tiled else "full"
    print(
        f"infer val: {len(items)} images | mode={mode} | score={score_thr} | min_area={min_area}",
        flush=True,
    )
    rows = _eval_rows(
        model,
        checkpoint,
        device,
        items,
        score_thr,
        min_area,
        tiled,
        tile_size,
        tile_overlap,
        desc="infer",
    )
    summary = _summary(rows)
    out_dir = config.edgeattn_path
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "val_metrics_edgeattn_tiled.json" if tiled else "val_metrics_edgeattn.json"
    (out_dir / name).write_text(
        json.dumps(
            {
                "config": {
                    "weights": str(weights),
                    "epoch": checkpoint.get("epoch"),
                    "arch": checkpoint.get("arch"),
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
    print("Home PQ (EdgeAttNet):", round(summary["pq"], 4), summary)
    _save_worst(
        rows,
        items,
        model,
        checkpoint,
        device,
        score_thr,
        min_area,
        viz_worst,
        tiled,
        tile_size,
        tile_overlap,
    )
    if submit:
        paths = sorted(
            p
            for p in config.test_images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if limit > 0:
            paths = paths[:limit]
        csv_rows = []
        for path in tqdm(paths, desc="submit test"):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise SystemExit(f"Failed to read image: {path}")
            labels = _predict_labels(
                model,
                gray,
                device,
                checkpoint,
                score_thr,
                min_area,
                tiled,
                tile_size,
                tile_overlap,
            )
            for i, mask in enumerate(instances_from_labels(labels), start=1):
                csv_rows.append(
                    {
                        "filament_id": f"{path.stem}_{i}",
                        "segmentation_rle": mask_to_rle_counts(mask),
                    }
                )
        _write_csv(config.submission_edgeattn_path, csv_rows)
