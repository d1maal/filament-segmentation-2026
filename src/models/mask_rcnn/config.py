from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaskRCNNConfig:
    num_classes: int = 2
    image_size: int = 1024

    epochs: int = 20
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.05

    score_threshold: float = 0.30
    mask_threshold: float = 0.50
    min_area: int = 80
    max_detections: int = 300

    rpn_pre_nms_top_n_train: int = 3000
    rpn_pre_nms_top_n_test: int = 3000
    rpn_post_nms_top_n_train: int = 1500
    rpn_post_nms_top_n_test: int = 1500
    rpn_nms_thresh: float = 0.70

    output_dir: Path = Path("outputs") / "mask_rcnn"


config = MaskRCNNConfig()
