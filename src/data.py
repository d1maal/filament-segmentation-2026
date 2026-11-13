from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from src.config import config

IMAGE_SIZE = 2048
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
HP_BLUR_SIGMA = 5.0
HP_SCALE = 2.0
HP_OFFSET = 128.0
VAL_FRAC = 0.2
VAL_SEED = 42


def three_channel(image: np.ndarray) -> np.ndarray:
    """Преобразование grayscale-изображения в трёхканальное: исходное, усиленное CLAHE и high-pass слой, склеенные по глубине.
    params:
        image: входное изображение
    returns:
        трёхканальный массив размером (H, W, 3)
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), int(HP_BLUR_SIGMA))
    hp = np.clip(
        (gray.astype(np.int16) - blur.astype(np.int16)) * HP_SCALE + HP_OFFSET,
        0,
        255,
    ).astype(np.uint8)
    return np.stack([gray, clahe, hp], axis=-1)


def load_coco(path: Path = config.train_annotations) -> COCO:
    """Загрузка COCO-аннотаций из JSON-файла.
    params:
        path: путь к файлу аннотаций
    returns:
        объект COCO с загруженными аннотациями
    """
    return COCO(str(path))


def image_stem(file_name: str) -> str:
    """Извлечение имени файла без расширения.
    params:
        file_name: имя файла в формате COCO
    returns:
        имя файла без расширения
    """
    return Path(file_name).stem


def resolve_image(images_dir: Path, file_name: str) -> Path | None:
    """Поиск реального файла изображения в папке.
    params:
        images_dir: папка с изображениями
        file_name: имя файла из COCO
    returns:
        путь к найденному файлу или None, если он не найден
    """
    path = images_dir / file_name
    if path.exists():
        return path
    hits = list(images_dir.glob(f"{Path(file_name).stem}.*"))
    return hits[0] if hits else None


def decode_ann_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Декодирование маски сегментации из аннотации COCO в бинарный массив нужного размера.
    params:
        ann: аннотация COCO
        height: высота изображения
        width: ширина изображения
    returns:
        бинарная маска размером (height, width)
    """
    seg = ann.get("segmentation")
    empty: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    if not seg:
        return empty
    if isinstance(seg, dict) and "counts" in seg:
        rle = (
            seg
            if not isinstance(seg.get("counts"), list)
            else mask_utils.frPyObjects(seg, height, width)
        )
        mask = mask_utils.decode(rle)
    else:
        rles = mask_utils.frPyObjects(seg, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def instances_for_image(coco: COCO, image_id: int) -> list[np.ndarray]:
    """Сбор всех масок-инстансов заданного изображения из аннотаций COCO.
    params:
        coco: загруженный объект COCO
        image_id: идентификатор изображения
    returns:
        список бинарных масок инстансов
    """
    info = coco.loadImgs([image_id])[0]
    h, w = int(info.get("height", IMAGE_SIZE)), int(info.get("width", IMAGE_SIZE))
    masks = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        mask = decode_ann_mask(ann, h, w)
        if mask.any():
            masks.append(mask)
    return masks


def mask_to_rle_counts(mask: np.ndarray) -> str:
    """Кодирование бинарной маски в строку RLE-счётчиков в формате pycocotools.
    params:
        mask: бинарная маска
    returns:
        строка RLE counts
    """
    rle = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def ensure_val_split(
    coco: COCO, frac: float = VAL_FRAC, seed: int = VAL_SEED
) -> set[str]:
    """Обеспечение существования фиксированного валидационного сплита.
    params:
        coco: загруженный объект COCO
        frac: доля изображений, отдаваемая в валидацию
        seed: сид генератора случайных чисел
    returns:
        множество имён валидационных изображений
    """
    for path in (config.val_ids_path, config.outputs_path / "val_ids.txt"):
        if path.exists():
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    stems = sorted(image_stem(coco.imgs[i]["file_name"]) for i in coco.imgs)
    rng = np.random.RandomState(seed)
    n = max(1, round(len(stems) * frac))
    chosen = sorted(rng.choice(stems, size=min(n, len(stems)), replace=False).tolist())
    out = config.val_ids_path
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    except OSError:
        out = config.outputs_path / "val_ids.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return set(chosen)
