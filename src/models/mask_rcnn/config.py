from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaskRCNNConfig:
    num_classes: int = 2
    architecture: str = "v3"
    image_size: int = 1344
    epochs: int = 42
    batch_size: int = 1
    lr: float = 1.5e-4
    min_lr: float = 5e-6
    warmup_epochs: int = 2
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2

    score_threshold: float = 0.55
    mask_threshold: float = 0.50
    min_area: int = 150
    max_detections: int = 200

    rpn_pre_nms_top_n_train: int = 3000
    rpn_pre_nms_top_n_test: int = 2000
    rpn_post_nms_top_n_train: int = 1500
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.70

    box_nms_thresh: float = 0.50
    box_fg_iou_thresh: float = 0.55
    box_bg_iou_thresh: float = 0.45
    box_positive_fraction: float = 0.20

    mask_roi_size: int = 32
    mask_head_depth: int = 6
    mask_dice_weight: float = 0.50

    anchor_aspect_ratios: tuple[float, ...] = (
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    )

    deduplicate: bool = True
    duplicate_mode: str = "weighted"
    mask_nms_iou: float = 0.60
    containment_threshold: float = 0.90

    stability_delta: float = 0.05
    min_mask_confidence: float = 0.0
    min_stability: float = 0.0
    quality_score_threshold: float = 0.0
    mask_confidence_power: float = 0.50
    stability_power: float = 0.25

    enforce_disjoint: bool = True
    overlap_score_weight: float = 0.35
    min_retained_fraction: float = 0.10

    min_tta_votes: int = 1
    single_view_score_threshold: float = 0.75
    tta_transforms_light: tuple[str, ...] = (
        "identity",
        "hflip",
        "vflip",
        "rot180",
    )
    tta_transforms_full: tuple[str, ...] = (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "hflip",
        "vflip",
        "transpose",
        "antitranspose",
    )

    autopsy_score_floor: float = 0.25
    autopsy_overlays: int = 20

    full_eval_start: int = 18
    full_eval_every: int = 6
    swa_last_n: int = 5
    checkpoint_keep_last: int = 5

    output_dir: Path = Path("outputs") / "mask_rcnn"


config = MaskRCNNConfig()
