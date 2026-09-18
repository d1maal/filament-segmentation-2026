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


@torch.no_grad()
def predict_masks(
    model,
    image,
    device,
    score_threshold=None,
    mask_threshold=None,
    min_area=None,
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

    output = model([image.to(device)])[0]
    result = []

    for score, mask in zip(output["scores"], output["masks"]):
        if float(score) < score_threshold:
            continue

        binary = (
            mask[0].detach().cpu().numpy() >= mask_threshold
        ).astype(np.uint8)

        if int(binary.sum()) < min_area:
            continue

        result.append(binary)

    return result
