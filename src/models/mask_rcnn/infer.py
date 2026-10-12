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


def _logit(x):
    x = np.clip(x, 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def prepare_output(output, score_floor: float = 0.0, inverse=None):
    scores = output["scores"].detach().cpu().numpy().astype(np.float32)
    keep = np.flatnonzero(scores >= float(score_floor))
    prepared = []
    for idx in keep:
        prob = output["masks"][int(idx), 0].detach().cpu().numpy().astype(np.float32)
        if inverse is not None:
            prob = inverse(prob)
        prepared.append({"score": float(scores[idx]), "prob": prob})
    return prepared


def _candidate_metrics(prob, score, mask_threshold, stability_delta, confidence_power, stability_power):
    binary = prob >= mask_threshold
    area = int(binary.sum())
    if area == 0:
        return area, 0.0, 0.0, 0.0
    mask_confidence = float(prob[binary].mean())
    low_thr = max(0.01, mask_threshold - stability_delta)
    high_thr = min(0.99, mask_threshold + stability_delta)
    low_area = int((prob >= low_thr).sum())
    high_area = int((prob >= high_thr).sum())
    stability = float(high_area / low_area) if low_area else 0.0
    quality = float(score)
    quality *= max(mask_confidence, 1e-6) ** float(confidence_power)
    quality *= max(stability, 1e-6) ** float(stability_power)
    return area, mask_confidence, stability, quality


def _deduplicate(candidates, mask_threshold, mode, iou_threshold, containment_threshold):
    if len(candidates) <= 1 or mode == "none":
        return candidates
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["quality"], reverse=True)
    kept = []
    for idx in order:
        candidate = dict(candidates[idx])
        binary = candidate["prob"] >= mask_threshold
        hit = -1
        for j, existing in enumerate(kept):
            existing_binary = existing["prob"] >= mask_threshold
            iou, containment = _mask_overlap(binary, existing_binary)
            if iou >= iou_threshold or containment >= containment_threshold:
                hit = j
                break
        if hit < 0:
            candidate["fusion_weight"] = max(candidate["quality"], 1e-6)
            kept.append(candidate)
            continue
        if mode == "suppress":
            continue
        if mode == "merge":
            kept[hit]["prob"] = np.maximum(kept[hit]["prob"], candidate["prob"])
            kept[hit]["score"] = max(kept[hit]["score"], candidate["score"])
            kept[hit]["quality"] = max(kept[hit]["quality"], candidate["quality"])
            continue
        if mode == "weighted":
            old_weight = float(kept[hit].get("fusion_weight", max(kept[hit]["quality"], 1e-6)))
            new_weight = max(float(candidate["quality"]), 1e-6)
            total = old_weight + new_weight
            kept[hit]["prob"] = (
                kept[hit]["prob"] * old_weight + candidate["prob"] * new_weight
            ) / total
            kept[hit]["fusion_weight"] = total
            kept[hit]["score"] = max(kept[hit]["score"], candidate["score"])
            kept[hit]["quality"] = max(kept[hit]["quality"], candidate["quality"])
            continue
        raise ValueError(f"Unknown duplicate mode: {mode}")
    return kept


def _assign_pixels(candidates, mask_threshold, overlap_score_weight):
    if not candidates:
        return []
    h, w = candidates[0]["prob"].shape
    dtype = np.int16 if len(candidates) < 32767 else np.int32
    owner = np.full((h, w), -1, dtype=dtype)
    best = np.full((h, w), -np.inf, dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        prob = candidate["prob"].astype(np.float32, copy=False)
        active = prob >= mask_threshold
        instance_quality = float(np.clip(candidate["quality"], 1e-5, 1.0 - 1e-5))
        value = _logit(prob) + float(overlap_score_weight) * float(_logit(instance_quality))
        update = active & (value > best)
        owner[update] = idx
        best[update] = value[update]
    return [(owner == idx).astype(np.uint8) for idx in range(len(candidates))]


def _make_disjoint(candidates, mask_threshold, min_area, overlap_score_weight):
    masks = _assign_pixels(candidates, mask_threshold, overlap_score_weight)
    keep = [i for i, mask in enumerate(masks) if int(mask.sum()) >= min_area]
    if len(keep) != len(masks):
        candidates = [candidates[i] for i in keep]
        masks = _assign_pixels(candidates, mask_threshold, overlap_score_weight)
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
    stability_delta=None,
    min_mask_confidence=None,
    min_stability=None,
    quality_score_threshold=None,
    mask_confidence_power=None,
    stability_power=None,
    enforce_disjoint=None,
    overlap_score_weight=None,
):
    score_threshold = config.score_threshold if score_threshold is None else float(score_threshold)
    mask_threshold = config.mask_threshold if mask_threshold is None else float(mask_threshold)
    min_area = config.min_area if min_area is None else int(min_area)
    deduplicate = config.deduplicate if deduplicate is None else bool(deduplicate)
    duplicate_mode = config.duplicate_mode if duplicate_mode is None else str(duplicate_mode)
    mask_nms_iou = config.mask_nms_iou if mask_nms_iou is None else float(mask_nms_iou)
    containment_threshold = config.containment_threshold if containment_threshold is None else float(containment_threshold)
    stability_delta = config.stability_delta if stability_delta is None else float(stability_delta)
    min_mask_confidence = config.min_mask_confidence if min_mask_confidence is None else float(min_mask_confidence)
    min_stability = config.min_stability if min_stability is None else float(min_stability)
    quality_score_threshold = config.quality_score_threshold if quality_score_threshold is None else float(quality_score_threshold)
    mask_confidence_power = config.mask_confidence_power if mask_confidence_power is None else float(mask_confidence_power)
    stability_power = config.stability_power if stability_power is None else float(stability_power)
    enforce_disjoint = config.enforce_disjoint if enforce_disjoint is None else bool(enforce_disjoint)
    overlap_score_weight = config.overlap_score_weight if overlap_score_weight is None else float(overlap_score_weight)

    candidates = []
    for item in prepared:
        score = float(item["score"])
        if score < score_threshold:
            continue
        prob = item["prob"].astype(np.float32, copy=False)
        area, confidence, stability, quality = _candidate_metrics(
            prob,
            score,
            mask_threshold,
            stability_delta,
            mask_confidence_power,
            stability_power,
        )
        if area < min_area:
            continue
        if confidence < min_mask_confidence:
            continue
        if stability < min_stability:
            continue
        if quality < quality_score_threshold:
            continue
        candidates.append(
            {
                "prob": prob,
                "score": score,
                "quality": quality,
                "mask_confidence": confidence,
                "stability": stability,
            }
        )

    if deduplicate:
        candidates = _deduplicate(
            candidates,
            mask_threshold,
            duplicate_mode,
            mask_nms_iou,
            containment_threshold,
        )

    if enforce_disjoint:
        return _make_disjoint(
            candidates,
            mask_threshold,
            min_area,
            overlap_score_weight,
        )

    return [
        (candidate["prob"] >= mask_threshold).astype(np.uint8)
        for candidate in candidates
        if int((candidate["prob"] >= mask_threshold).sum()) >= min_area
    ]


def postprocess_output(output, **kwargs):
    score_threshold = kwargs.get("score_threshold")
    floor = config.score_threshold if score_threshold is None else float(score_threshold)
    prepared = prepare_output(output, score_floor=floor)
    return postprocess_prepared(prepared, **kwargs)


def _apply_tta(image, name):
    if name == "identity":
        return image, lambda x: x
    if name == "hflip":
        return torch.flip(image, dims=[2]), lambda x: np.flip(x, axis=1).copy()
    if name == "vflip":
        return torch.flip(image, dims=[1]), lambda x: np.flip(x, axis=0).copy()
    if name == "rot180":
        return torch.rot90(image, 2, dims=[1, 2]), lambda x: np.rot90(x, 2).copy()
    raise ValueError(f"Unknown TTA transform: {name}")


@torch.no_grad()
def prepare_tta(model, image, device, transforms=None, score_floor=0.0):
    transforms = tuple(transforms or config.tta_transforms)
    prepared = []
    image = image.to(device)
    for name in transforms:
        transformed, inverse = _apply_tta(image, name)
        output = model([transformed])[0]
        prepared.extend(prepare_output(output, score_floor=score_floor, inverse=inverse))
    return prepared


@torch.no_grad()
def predict_masks(model, image, device, tta=False, **kwargs):
    score_threshold = kwargs.get("score_threshold")
    floor = config.score_threshold if score_threshold is None else float(score_threshold)
    if tta:
        transforms = config.tta_transforms if tta is True else tuple(tta)
        prepared = prepare_tta(model, image, device, transforms=transforms, score_floor=floor)
        return postprocess_prepared(prepared, **kwargs)
    output = model([image.to(device)])[0]
    return postprocess_output(output, **kwargs)
