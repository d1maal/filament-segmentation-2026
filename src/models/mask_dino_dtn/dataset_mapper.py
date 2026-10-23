"""3-канальный DatasetMapper для MaskDINO (gray + CLAHE + high-pass)."""

from __future__ import annotations

import cv2
import numpy as np
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen

from src.data import three_channel
from src.models.mask_dino_dtn.maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    COCOInstanceNewBaselineDatasetMapper,
    build_transform_gen,
)

__all__ = ["MagfiloThreeChannelDatasetMapper"]


def _build_test_transform_gen(cfg) -> list[TransformGen]:
    """Тестовые трансформы: resize по короткой стороне, без кропа и флипа.

    params:
        cfg: конфиг detectron2
    returns:
        список тестовых трансформов
    """
    return [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)]


class MagfiloThreeChannelDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    """Читает H-alpha снимок как 3 канала (gray + CLAHE + high-pass)."""

    @classmethod
    def from_config(cls, cfg, is_train: bool = True) -> dict:
        """Собирает трансформы и параметры маппера.

        params:
            cfg: конфиг detectron2
            is_train: True — LSJ-аугментации, False — только resize
        returns:
            словарь аргументов для __init__
        """
        tfm_gens = (
            build_transform_gen(cfg, is_train) if is_train else _build_test_transform_gen(cfg)
        )
        return {"is_train": is_train, "tfm_gens": tfm_gens, "image_format": cfg.INPUT.FORMAT}

    def _load_image(self, dataset_dict: dict) -> np.ndarray:
        """Читает grayscale-снимок и строит 3 канала (gray + CLAHE + high-pass).

        params:
            dataset_dict: словарь detectron2 с полем file_name
        returns:
            массив (H, W, 3) uint8
        """
        gray = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_GRAYSCALE)
        return three_channel(gray)
