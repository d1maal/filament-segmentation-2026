from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import config


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