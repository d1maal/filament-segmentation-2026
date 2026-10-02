from __future__ import annotations

import cv2
import numpy as np
import torch
from sahi.models.base import DetectionModel
from sahi.postprocess.combine import PostprocessPredictions
from sahi.postprocess.utils import merge_object_prediction_pair
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

from src.data import three_channel
from src.models.m2f.inference import _instances_from_result
from src.models.m2f.tiling import _mask_similarity


class Mask2FormerSahiModel(DetectionModel):
    """Обёртка Mask2FormerForUniversalSegmentation для SAHI DetectionModel."""

    def __init__(self, model, processor, device, score_thr: float = 0.3) -> None:
        self._processor = processor
        self._torch_model = model
        self._device = device
        self._score_thr = score_thr
        super().__init__(
            device=str(device),
            confidence_threshold=score_thr,
            category_mapping={0: "filament"},
            load_at_init=False,
        )
        self.model = model

    def load_model(self) -> None:
        """Требуется базовым классом; модель уже загружена на этапе конструирования."""

    @torch.no_grad()
    def perform_inference(self, image: np.ndarray) -> None:
        """Прогоняет один слайс и сохраняет список (маска, скор)"""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        inputs = {
            k: v.to(self._device)
            for k, v in self._processor(
                images=three_channel(gray), return_tensors="pt"
            ).items()
        }
        result = self._processor.post_process_instance_segmentation(
            self._torch_model(**inputs),
            threshold=self._score_thr,
            target_sizes=[(h, w)],
            return_binary_maps=True,
        )[0]
        self._original_predictions = _instances_from_result(result, self._score_thr, 0)

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list=None, full_shape_list=None
    ) -> None:
        """Конвертирует бинарные маски слайса в ObjectPrediction."""
        from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift = [int(v) for v in shift_amount_list[0]]
        full_shape = [int(v) for v in full_shape_list[0]] if full_shape_list else None
        self._object_prediction_list_per_image = [
            [
                ObjectPrediction(
                    bbox=None,
                    segmentation=get_coco_segmentation_from_bool_mask(mask.astype(bool)),
                    category_id=0,
                    category_name="filament",
                    shift_amount=shift,
                    full_shape=full_shape,
                    score=float(score),
                )
                for mask, score in (self._original_predictions or [])
                if mask.any()
            ]
        ]


class MaskGreedyNMM(PostprocessPredictions):
    """Жадное слияние дублей по сходству масок."""

    def __call__(self, object_predictions: list[ObjectPrediction]) -> list[ObjectPrediction]:
        masks = [p.mask.bool_mask if p.mask is not None else None for p in object_predictions]
        order = sorted(
            range(len(object_predictions)),
            key=lambda i: object_predictions[i].score.value,
            reverse=True,
        )
        merged: list[ObjectPrediction] = []
        merged_masks: list[np.ndarray | None] = []
        for i in order:
            m = masks[i]
            if m is None:
                continue
            hit = False
            for j, (kept, km) in enumerate(zip(merged, merged_masks)):
                if km is not None and _mask_similarity(m, km) >= self.match_threshold:
                    new = merge_object_prediction_pair(kept, object_predictions[i])
                    merged[j] = new
                    merged_masks[j] = new.mask.bool_mask if new.mask is not None else None
                    hit = True
                    break
            if not hit:
                merged.append(object_predictions[i])
                merged_masks.append(m)
        return merged


@torch.no_grad()
def predict_masks_sahi(
    model,
    processor,
    image: np.ndarray,
    device,
    score_thr: float = 0.3,
    min_area: int = 80,
    slice_size: int = 1024,
    overlap: int = 256,
    merge_overlap: float = 0.5,
    postprocess_type: str = "MASKNMM",
    postprocess_match_metric: str = "IOS",
    perform_standard_pred: bool = True,
) -> list[np.ndarray]:
    """Sliced-инференс через SAHI; по выходу совместим с predict_masks_tiled.

    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        image: grayscale-изображение (или BGR)
        device: устройство выполнения вычислений
        score_thr: минимальный скор детекции
        min_area: минимальная площадь маски после слияния
        slice_size: сторона слайса
        overlap: перекрытие слайсов в пикселях
        merge_overlap: порог слияния дублей
        postprocess_type: 'MASKNMM' (по маскам), либо 'GREEDYNMM'/'NMM'/'NMS'
        postprocess_match_metric: 'IOU'/'IOS' для стандартных постпроцессоров SAHI
        perform_standard_pred: добавить предсказание по полному кадру
    returns:
        список бинарных масок инстансов
    """
    from sahi.predict import POSTPROCESS_NAME_TO_CLASS, get_sliced_prediction

    POSTPROCESS_NAME_TO_CLASS["MASKNMM"] = MaskGreedyNMM
    detection_model = Mask2FormerSahiModel(model, processor, device, score_thr)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap / slice_size,
        overlap_width_ratio=overlap / slice_size,
        perform_standard_pred=perform_standard_pred,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=merge_overlap,
        postprocess_class_agnostic=True,
        verbose=0,
    )
    masks: list[np.ndarray] = []
    for pred in result.object_prediction_list:
        if pred.mask is None:
            continue
        m = pred.mask.bool_mask
        if int(m.sum()) >= min_area:
            masks.append(m.astype(np.uint8))
    return masks
