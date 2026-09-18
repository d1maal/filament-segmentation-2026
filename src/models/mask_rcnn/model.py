from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

from .config import config


def build_model(image_size: int | None = None):
    size = int(image_size or config.image_size)

    # Filaments are often thin/elongated, so include extreme aspect ratios.
    # Passing the generator at construction time keeps the RPN head consistent
    # with the number of anchors per location.
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    )

    return maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=config.num_classes,
        min_size=size,
        max_size=size,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=config.rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=config.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=config.rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=config.rpn_post_nms_top_n_test,
        rpn_nms_thresh=config.rpn_nms_thresh,
        box_score_thresh=0.05,
        box_detections_per_img=config.max_detections,
    )
