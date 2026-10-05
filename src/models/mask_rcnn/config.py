from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaskRCNNConfig:
    num_classes: int = 2
    architecture: str = "v2"
    image_size: int = 1280
    epochs: int = 30
    batch_size: int = 1
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2
    score_threshold: float = 0.50
    mask_threshold: float = 0.50
    min_area: int = 150
    max_detections: int = 200
    rpn_pre_nms_top_n_train: int = 3000
    rpn_pre_nms_top_n_test: int = 2000
    rpn_post_nms_top_n_train: int = 1500
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.70
    box_nms_thresh: float = 0.50
    mask_roi_size: int = 28
    anchor_aspect_ratios: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    deduplicate: bool = True
    duplicate_mode: str = "merge"
    mask_nms_iou: float = 0.60
    containment_threshold: float = 0.90
    enforce_disjoint: bool = True
    overlap_score_power: float = 0.50
    output_dir: Path = Path("outputs") / "mask_rcnn"


config = MaskRCNNConfig()
