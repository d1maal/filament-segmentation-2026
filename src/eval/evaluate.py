from __future__ import annotations

import json

import cv2
import numpy as np
from tqdm import tqdm

from src.baseline import predict_instances
from src.config import config
from src.data import image_stem, instances_for_image, mask_to_rle_counts
from src.eval.submission import _write_csv
from src.eval.val_items import _val_items
from src.io.sweep_log import _append_sweep
from src.io.visualization import _save_worst
from src.metrics import aggregate, panoptic_quality
from src.models.m2f import load_checkpoint as _load_model
from src.models.m2f.inference import _predict, predict_masks_grid


def _summary(rows: list[dict]) -> dict:
    """Агрегирует PQ и добавляет SQ/RQ/precision/recall.
    params:
        rows: покартинные метрики
    returns:
        сводный словарь
    """
    return aggregate(rows)


def cmd_baseline(submit: bool, viz: int) -> None:
    """Считает Otsu-байзлайн на валидации, опционально рисует viz и пишет submission CSV.
    params:
        submit: флаг генерации сабмишн CSV
        viz: число первых изображений для визуализации (0 — без неё)
    """
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


def cmd_check() -> None:
    """Запускает самопроверку пайплайна: PQ, split, трёхканальность и RLE; данные не требуются."""
    from src.data import mask_to_rle_counts, three_channel
    from src.metrics import panoptic_quality

    m: np.ndarray = np.zeros((32, 32), np.uint8)
    m[5:15, 5:20] = 1
    assert panoptic_quality([m], [m.copy()])["pq"] == 1.0
    a, b = np.zeros_like(m), np.zeros_like(m)
    a[5:15, 5:12] = 1
    b[5:15, 14:20] = 1
    assert panoptic_quality([m], [a, b])["splits"] == 1
    rng = np.random.default_rng(7)
    for _ in range(25):
        gt = [
            rng.random((64, 64)) < rng.uniform(0.01, 0.2)
            for _ in range(rng.integers(0, 5))
        ]
        pred = [
            rng.random((64, 64)) < rng.uniform(0.01, 0.2)
            for _ in range(rng.integers(0, 5))
        ]
        got = panoptic_quality(gt, pred)
        ref = np.zeros((len(gt), len(pred)))
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                inter = int(np.logical_and(g > 0, p > 0).sum())
                union = int(np.logical_or(g > 0, p > 0).sum())
                ref[i, j] = inter / union if union else 0.0
        used_g, used_p, matches = set(), set(), []
        if ref.size:
            for idx in np.argsort(-ref, axis=None):
                gi, pj = map(int, np.unravel_index(int(idx), ref.shape))
                if ref[gi, pj] < 0.5:
                    break
                if gi in used_g or pj in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pj)
                matches.append(float(ref[gi, pj]))
        tp = len(matches)
        denom = tp + 0.5 * (len(pred) - tp) + 0.5 * (len(gt) - tp)
        ref_pq = sum(matches) / denom if denom else 0.0
        assert abs(got["pq"] - ref_pq) < 1e-9
        assert got["tp"] == tp
        assert got["fp"] == len(pred) - tp
        assert got["fn"] == len(gt) - tp
    rgb = three_channel(np.full((64, 64), 120, np.uint8))
    assert rgb.shape == (64, 64, 3)
    assert mask_to_rle_counts(m)
    print("check ok")


def cmd_infer(
    submit: bool,
    weights,
    score_thr: float,
    min_area: int,
    viz_worst: int,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    merge_overlap: float,
    limit: int,
    sahi: bool = False,
    tta: bool = False,
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
        tta: test-time augmentation
    """
    weights, checkpoint, processor, model, device = _load_model(weights)
    items = _val_items()
    if limit > 0:
        items = items[:limit]
    rows = []
    mode = "tta" if tta else ("sahi" if sahi else ("tiled" if tiled else "full"))
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
            tiled, tile_size, tile_overlap, merge_overlap, sahi, tta,
        )
        metrics = panoptic_quality(instances_for_image(coco, info["id"]), pred)
        metrics["stem"] = image_stem(info["file_name"])
        rows.append(metrics)
    summary = _summary(rows)
    config.models_path.mkdir(parents=True, exist_ok=True)
    name = "val_metrics_m2f_tiled.json" if (tiled or sahi or tta) else "val_metrics_m2f.json"
    (config.models_path / name).write_text(
        json.dumps(
            {
                "config": {
                    "weights": str(weights),
                    "epoch": checkpoint.get("epoch"),
                    "score_thr": score_thr,
                    "min_area": min_area,
                    "tiled": tiled,
                    "tta": tta,
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
        tiled, tile_size, tile_overlap, merge_overlap, sahi, tta,
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
                tiled, tile_size, tile_overlap, merge_overlap, sahi, tta,
            )
            for i, mask in enumerate(pred, start=1):
                csv_rows.append(
                    {"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle_counts(mask)}
                )
        _write_csv(config.submission_model_path, csv_rows)


def cmd_sweep(
    weights, score_thrs: list[float], min_areas: list[int], limit: int, write_log: bool
) -> None:
    """Сетка score_thr × min_area на одном forward на картинку.
    params:
        weights: чекпоинт
        score_thrs: пороги скора
        min_areas: пороги площади
        limit: ограничить val (0 — все)
        write_log: дописать experiments.md
    """
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