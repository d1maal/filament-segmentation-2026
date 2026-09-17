from __future__ import annotations

import csv
from pathlib import Path


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