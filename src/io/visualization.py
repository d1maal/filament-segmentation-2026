from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import config
from src.data import image_stem, instances_for_image
from src.eval.val_items import _val_items
from src.io.overlay import _draw_overlay
from src.metrics import panoptic_quality

BEST_WORST_TOP_K = 10


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
    from src.models.m2f.inference import _predict

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


def _save_best_worst(model, processor, device, run_dir: Path) -> None:
    """Прогоняет модель по всей валидации и сохраняет top-k худших/лучших снимков с overlay GT/pred.
    params:
        model: модель с весами best.pt
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        run_dir: папка эксперимента (outputs/m2f/experiments/<run_name>)
    """
    from src.models.m2f.inference import predict_masks

    results = []
    for coco, info, path in tqdm(_val_items(), desc="best/worst val"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_masks(model, processor, img, device)
        pq = panoptic_quality(instances_for_image(coco, info["id"]), pred)["pq"]
        results.append((pq, info, path, coco))
    results.sort(key=lambda r: r[0])
    for name, seq in (("bad", results[:BEST_WORST_TOP_K]), ("good", results[-BEST_WORST_TOP_K:])):
        folder = run_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for pq, info, path, coco in seq:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for m in instances_for_image(coco, info["id"]):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
            for m in predict_masks(model, processor, gray, device):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (0, 255, 0), 2)
            cv2.imwrite(str(folder / f"pq{pq:.3f}_{path.stem}.png"), vis)