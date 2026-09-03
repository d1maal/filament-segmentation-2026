from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 2048
OUTPUTS = Path("/kaggle/working/outputs") if Path("/kaggle/working").is_dir() else ROOT / "outputs"
VAL_IDS = ROOT / "splits" / "val_ids.txt"


def discover_data_root() -> Path:
    env = os.environ.get("MAGFILO_ROOT")
    if env:
        return Path(env)
    candidates = [
        ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026",
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/filament-segmentation-2026"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(p.parent.parent if p.parent.name == "train" else p.parent for p in kaggle_input.rglob("train_images"))
    for root in candidates:
        if (root / "train" / "train_images").is_dir():
            return root
    return candidates[0]


DATA_ROOT = discover_data_root()
TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
_ann = list((DATA_ROOT / "train").glob("*Annotations*.json"))
TRAIN_ANN = _ann[0] if _ann else DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TEST_IMAGES = DATA_ROOT / "test" / "test_images"


def three_channel(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    hp = np.clip((gray.astype(np.int16) - blur.astype(np.int16)) * 2 + 128, 0, 255).astype(np.uint8)
    return np.stack([gray, clahe, hp], axis=-1)


def load_coco(path: Path = TRAIN_ANN) -> COCO:
    return COCO(str(path))


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def resolve_image(images_dir: Path, file_name: str) -> Path | None:
    path = images_dir / file_name
    if path.exists():
        return path
    hits = list(images_dir.glob(f"{Path(file_name).stem}.*"))
    return hits[0] if hits else None


def decode_ann_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    empty = np.zeros((height, width), dtype=np.uint8)
    if not seg:
        return empty
    if isinstance(seg, dict) and "counts" in seg:
        rle = seg if not isinstance(seg.get("counts"), list) else mask_utils.frPyObjects(seg, height, width)
        mask = mask_utils.decode(rle)
    else:
        rles = mask_utils.frPyObjects(seg, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def instances_for_image(coco: COCO, image_id: int) -> list[np.ndarray]:
    info = coco.loadImgs([image_id])[0]
    h, w = int(info.get("height", IMAGE_SIZE)), int(info.get("width", IMAGE_SIZE))
    masks = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        mask = decode_ann_mask(ann, h, w)
        if mask.any():
            masks.append(mask)
    return masks


def mask_to_rle_counts(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def ensure_val_split(coco: COCO, frac: float = 0.2, seed: int = 42) -> set[str]:
    for path in (VAL_IDS, OUTPUTS / "val_ids.txt"):
        if path.exists():
            return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    stems = sorted(image_stem(coco.imgs[i]["file_name"]) for i in coco.imgs)
    rng = np.random.RandomState(seed)
    n = max(1, int(round(len(stems) * frac)))
    chosen = sorted(rng.choice(stems, size=min(n, len(stems)), replace=False).tolist())
    out = VAL_IDS
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    except OSError:
        out = OUTPUTS / "val_ids.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return set(chosen)
