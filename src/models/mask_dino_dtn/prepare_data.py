from __future__ import annotations

import json

from src.config import config
from src.data import ensure_val_split, image_stem, load_coco

OUT_DIR = config.outputs_path / "mask_dino_dtn"
CATEGORY_ID = 1

def _build_split(dataset: dict, val_stems: set[str], *, is_val: bool) -> dict:
    """Собирает COCO-словарь для train или val с целыми id и одной категорией.

    params:
        dataset: исходный COCO-словарь (images/annotations)
        val_stems: множество стемов валидационных изображений
        is_val: True — собрать val, False — train
    returns:
        словарь с categories/images/annotations
    """
    img_id_map: dict = {}  # старый (строковый) id -> новый int id
    images: list[dict] = []
    next_img_id = 1
    for img in dataset["images"]:
        if (image_stem(img["file_name"]) in val_stems) != is_val:
            continue
        img_id_map[img["id"]] = next_img_id
        images.append(
            {
                "id": next_img_id,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        )
        next_img_id += 1

    annotations: list[dict] = []
    next_ann_id = 1
    for ann in dataset["annotations"]:
        if ann["image_id"] not in img_id_map:
            continue
        annotations.append(
            {
                "id": next_ann_id,
                "image_id": img_id_map[ann["image_id"]],
                "category_id": CATEGORY_ID,
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "iscrowd": 0,
            }
        )
        next_ann_id += 1

    return {
        "categories": [{"id": CATEGORY_ID, "name": "filament", "supercategory": "filament"}],
        "images": images,
        "annotations": annotations,
    }

def main() -> None:
    """Генерирует оба JSON и печатает статистику."""
    coco = load_coco()
    dataset = coco.dataset
    val_stems = ensure_val_split(coco)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, is_val in (("magfilo_train", False), ("magfilo_val", True)):
        split = _build_split(dataset, val_stems, is_val=is_val)
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(split), encoding="utf-8")
        print(f"{out}: {len(split['images'])} images, {len(split['annotations'])} anns")


if __name__ == "__main__":
    main()