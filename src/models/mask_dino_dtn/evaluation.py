"""Home PQ эвалюатор для detectron2."""

from __future__ import annotations

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator

from src.data import decode_ann_mask
from src.metrics import aggregate, panoptic_quality


class HomePQEvaluator(DatasetEvaluator):
    """Считает Home PQ между GT-разметкой и предсказаниями модели.

    GT-маски декодируются на лету из аннотаций, чтобы не держать в памяти
    все маски val-сплита (2048x2048 каждая).

    Attributes:
        score_threshold: порог уверенности для предсказанных масок
    """

    def __init__(self, dataset_name: str, score_threshold: float = 0.25) -> None:
        self._dataset_name = dataset_name
        self._score_threshold = score_threshold
        self._rows: list[dict] = []
        self._gt: dict[int, tuple[int, int, list[dict]]] = {}

    def reset(self) -> None:
        """Сбрасывает накопленные строки и перечитывает GT-аннотации."""
        self._rows = []
        self._gt = self._load_gt()

    def _load_gt(self) -> dict[int, tuple[int, int, list[dict]]]:
        """Собирает аннотации по image_id из зарегистрированного датасета.

        returns:
            словарь image_id -> (height, width, список аннотаций)
        """
        gt: dict[int, tuple[int, int, list[dict]]] = {}
        for record in DatasetCatalog.get(self._dataset_name):
            gt[record["image_id"]] = (
                int(record["height"]),
                int(record["width"]),
                list(record.get("annotations", [])),
            )
        return gt

    def process(self, inputs: list[dict], outputs: list[dict]) -> None:
        """Считает PQ для каждого изображения батча и копит строки метрик.

        params:
            inputs: список словарей detectron2 (по одному на изображение)
            outputs: список результатов модели с полем instances
        """
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            gt_masks: list[np.ndarray] = []
            gt_h, gt_w = 0, 0
            if image_id in self._gt:
                gt_h, gt_w, annotations = self._gt[image_id]
                decoded = [decode_ann_mask(ann, gt_h, gt_w) for ann in annotations]
                gt_masks = [mask for mask in decoded if mask.any()]

            instances = output["instances"]
            pred_masks: list[np.ndarray] = []
            for mask, score in zip(instances.pred_masks, instances.scores):
                if float(score) < self._score_threshold:
                    continue
                pred = mask.cpu().numpy()
                if gt_h and pred.shape != (gt_h, gt_w):
                    pred = cv2.resize(
                        pred.astype(np.uint8), (gt_w, gt_h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                pred_masks.append(pred)
            self._rows.append(panoptic_quality(gt_masks, pred_masks))

    def evaluate(self) -> dict:
        """Агрегирует Home PQ по всем обработанным изображениям.

        returns:
            словарь с метриками PQ, TP, FP и FN
        """
        result = aggregate(self._rows)
        return {
            "PQ": result["pq"],
            "TP": result["tp"],
            "FP": result["fp"],
            "FN": result["fn"],
        }
