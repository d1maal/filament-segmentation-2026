from pathlib import Path

import numpy as np
import torch

from .config import config
from .model import build_model


def load_checkpoint(path: str | Path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        image_size = int(checkpoint.get("image_size", config.image_size))
        architecture = checkpoint.get("architecture", "v1")
        state_dict = checkpoint.get("model", checkpoint)
    else:
        image_size = config.image_size
        architecture = "v1"
        state_dict = checkpoint
    model = build_model(image_size=image_size, architecture=architecture)
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
    return float(inter / union), float(inter / min(area_a, area_b))


def prepare_output(output, score_floor: float = 0.0):
    scores = output["scores"].detach().cpu().numpy().astype(np.float32)
    keep = np.flatnonzero(scores >= float(score_floor))
    prepared_scores = []
    prepared_probs = []
    for idx in keep:
        prepared_scores.append(float(scores[idx]))
        prepared_probs.append(output["masks"][int(idx), 0].detach().cpu().numpy().astype(np.float16))
    return prepared_scores, prepared_probs


def _deduplicate(
    probs: list[np.ndarray],
    scores: list[float],
    mask_threshold: float,
    mode: str,
    iou_threshold: float,
    containment_threshold: float,
):
    if len(probs) <= 1 or mode == "none":
        return probs, scores
    order = sorted(range(len(probs)), key=lambda i: scores[i], reverse=True)
    kept_probs = []
    kept_scores = []
    for idx in order:
        prob = probs[idx].astype(np.float32, copy=False)
        score = scores[idx]
        binary = prob >= mask_threshold
        hit = -1
        for j, kept_prob in enumerate(kept_probs):
            kept_binary = kept_prob >= mask_threshold
            iou, containment = _mask_overlap(binary, kept_binary)
            if iou >= iou_threshold or containment >= containment_threshold:
                hit = j
                break
        if hit < 0:
            kept_probs.append(prob.copy())
            kept_scores.append(score)
        elif mode == "merge":
            kept_probs[hit] = np.maximum(kept_probs[hit], prob)
            kept_scores[hit] = max(kept_scores[hit], score)
    return kept_probs, kept_scores


def _assign_pixels(
    probs: list[np.ndarray],
    scores: list[float],
    mask_threshold: float,
    score_power: float,
):
    if not probs:
        return []
    h, w = probs[0].shape
    owner = np.full((h, w), -1, dtype=np.int16 if len(probs) < 32767 else np.int32)
    best = np.full((h, w), -np.inf, dtype=np.float32)
    for idx, (prob, score) in enumerate(zip(probs, scores)):
        p = prob.astype(np.float32, copy=False)
        active = p >= mask_threshold
        weight = float(score) ** float(score_power)
        update = active & (p * weight > best)
        owner[update] = idx
        best[update] = p[update] * weight
    return [(owner == idx).astype(np.uint8) for idx in range(len(probs))]


def _make_disjoint(
    probs: list[np.ndarray],
    scores: list[float],
    mask_threshold: float,
    min_area: int,
    score_power: float,
):
    masks = _assign_pixels(probs, scores, mask_threshold, score_power)
    keep = [i for i, mask in enumerate(masks) if int(mask.sum()) >= min_area]
    if len(keep) != len(masks):
        probs = [probs[i] for i in keep]
        scores = [scores[i] for i in keep]
        masks = _assign_pixels(probs, scores, mask_threshold, score_power)
    return [mask for mask in masks if int(mask.sum()) >= min_area]


def postprocess_prepared(
    prepared,
    score_threshold=None,
    mask_threshold=None,
    min_area=None,
    deduplicate=None,
    duplicate_mode=None,
    mask_nms_iou=None,
    containment_threshold=None,
    enforce_disjoint=None,
    overlap_score_power=None,
):
    score_threshold = config.score_threshold if score_threshold is None else float(score_threshold)
    mask_threshold = config.mask_threshold if mask_threshold is None else float(mask_threshold)
    min_area = config.min_area if min_area is None else int(min_area)
    deduplicate = config.deduplicate if deduplicate is None else bool(deduplicate)
    duplicate_mode = config.duplicate_mode if duplicate_mode is None else str(duplicate_mode)
    mask_nms_iou = config.mask_nms_iou if mask_nms_iou is None else float(mask_nms_iou)
    containment_threshold = config.containment_threshold if containment_threshold is None else float(containment_threshold)
    enforce_disjoint = config.enforce_disjoint if enforce_disjoint is None else bool(enforce_disjoint)
    overlap_score_power = config.overlap_score_power if overlap_score_power is None else float(overlap_score_power)
    scores, probs = prepared
    selected_probs = []
    selected_scores = []
    for score, prob in zip(scores, probs):
        if score < score_threshold:
            continue
        binary = prob >= mask_threshold
        if int(binary.sum()) < min_area:
            continue
        selected_probs.append(prob)
        selected_scores.append(score)
    if deduplicate:
        selected_probs, selected_scores = _deduplicate(
            selected_probs,
            selected_scores,
            mask_threshold,
            duplicate_mode,
            mask_nms_iou,
            containment_threshold,
        )
    if enforce_disjoint:
        return _make_disjoint(
            selected_probs,
            selected_scores,
            mask_threshold,
            min_area,
            overlap_score_power,
        )
    return [
        (prob >= mask_threshold).astype(np.uint8)
        for prob in selected_probs
        if int((prob >= mask_threshold).sum()) >= min_area
    ]


def postprocess_output(output, **kwargs):
    score_threshold = kwargs.get("score_threshold")
    floor = config.score_threshold if score_threshold is None else float(score_threshold)
    prepared = prepare_output(output, score_floor=floor)
    return postprocess_prepared(prepared, **kwargs)


@torch.no_grad()
def predict_masks(model, image, device, **kwargs):
    output = model([image.to(device)])[0]
    return postprocess_output(output, **kwargs)
