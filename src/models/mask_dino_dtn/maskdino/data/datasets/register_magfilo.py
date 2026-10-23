"""Регистрация датасета MAGFiLO (filament) в detectron2."""

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from src.config import config

IMAGE_ROOT = str(config.train_images)
JSON_DIR = config.outputs_path / "mask_dino_dtn"
_DATASETS = ("magfilo_train", "magfilo_val")


def register_magfilo() -> None:
    """Регистрирует magfilo_train и magfilo_val как COCO instance-датасеты."""
    
    for name in _DATASETS:
        json_file = str(JSON_DIR / f"{name}.json")
        DatasetCatalog.register(
            name,
            lambda j=json_file, r=IMAGE_ROOT, n=name: load_coco_json(j, r, n),
        )
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=IMAGE_ROOT,
            evaluator_type="magfilo",
        )


register_magfilo()
