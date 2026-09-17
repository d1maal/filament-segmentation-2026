from __future__ import annotations

from src.config import config
from src.data import ensure_val_split, image_stem, load_coco, resolve_image


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