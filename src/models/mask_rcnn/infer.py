from pathlib import Path

import numpy as np
import torch

from .config import config
from .model import build_model


def load_checkpoint(path: str | Path, device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    image_size = (
        int(checkpoint.get("image_size", config.image_size))
        if isinstance(checkpoint, dict)
        else config.image_size
    )

    model = build_model(image_size=image_size)

    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


def _mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    area_a = int(a.sum())
    area_b = int(b.sum())
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0

    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0

    union = area_a + area_b - inter
    iou = inter / union if union else 0.0
    containment = inter / min(area_a, area_b)
    return float(iou), float(containment)


def suppress_duplicate_masks(
    masks: list[np.ndarray],
    scores: list[float],
    iou_threshold: float = 0.60,
    containment_threshold: float = 0.90,
) -> tuple[list[np.ndarray], list[float]]:
    """Score-ordered mask NMS with an additional containment test."""
    if len(masks) <= 1:
        return masks, scores

    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []

    for idx in order:
        candidate = masks[idx]
        duplicate = False

        for kept in kept_masks:
            iou, containment = _mask_overlap(candidate, kept)
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break

        if not duplicate:
            kept_masks.append(candidate)
            kept_scores.append(scores[idx])

    return kept_masks, kept_scores


def postprocess_output(
    output,
    score_threshold=None,
    mask_threshold=None,
    min_area=None,
    deduplicate=None,
    mask_nms_iou=None,
    containment_threshold=None,
):
    score_threshold = (
        config.score_threshold
        if score_threshold is None
        else float(score_threshold)
    )
    mask_threshold = (
        config.mask_threshold
        if mask_threshold is None
        else float(mask_threshold)
    )
    min_area = config.min_area if min_area is None else int(min_area)
    deduplicate = config.deduplicate if deduplicate is None else bool(deduplicate)
    mask_nms_iou = (
        config.mask_nms_iou
        if mask_nms_iou is None
        else float(mask_nms_iou)
    )
    containment_threshold = (
        config.containment_threshold
        if containment_threshold is None
        else float(containment_threshold)
    )

    masks: list[np.ndarray] = []
    scores: list[float] = []

    for score, mask in zip(output["scores"], output["masks"]):
        score_value = float(score.detach().cpu())
        if score_value < score_threshold:
            continue

        binary = (
            mask[0].detach().cpu().numpy() >= mask_threshold
        ).astype(np.uint8)

        if int(binary.sum()) < min_area:
            continue

        masks.append(binary)
        scores.append(score_value)

    if deduplicate:
        masks, scores = suppress_duplicate_masks(
            masks,
            scores,
            iou_threshold=mask_nms_iou,
            containment_threshold=containment_threshold,
        )

    return masks


@torch.no_grad()
def predict_masks(
    model,
    image,
    device,
    score_threshold=None,
    mask_threshold=None,
    min_area=None,
    deduplicate=None,
    mask_nms_iou=None,
    containment_threshold=None,
):
    output = model([image.to(device)])[0]
    return postprocess_output(
        output,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        min_area=min_area,
        deduplicate=deduplicate,
        mask_nms_iou=mask_nms_iou,
        containment_threshold=containment_threshold,
    )
