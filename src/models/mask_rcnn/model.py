from functools import partial

import torch.nn as nn
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign

from .config import config


def _anchor_generator(ratios=None):
    ratios = tuple(float(x) for x in (ratios or config.anchor_aspect_ratios))
    return AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=(ratios,) * 5,
    )


def _build_v1(size: int):
    return maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=config.num_classes,
        min_size=size,
        max_size=size,
        rpn_anchor_generator=_anchor_generator((0.25, 0.5, 1.0, 2.0, 4.0)),
        rpn_pre_nms_top_n_train=3000,
        rpn_pre_nms_top_n_test=3000,
        rpn_post_nms_top_n_train=1500,
        rpn_post_nms_top_n_test=1500,
        rpn_nms_thresh=0.70,
        box_score_thresh=0.05,
        box_nms_thresh=0.50,
        box_detections_per_img=300,
    )


def _build_v2(size: int):
    norm_layer = partial(nn.GroupNorm, 32)
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=norm_layer,
        trainable_layers=5,
    )
    anchors = _anchor_generator()
    rpn_head = RPNHead(
        backbone.out_channels,
        anchors.num_anchors_per_location()[0],
        conv_depth=2,
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7),
        [256, 256, 256, 256],
        [1024],
        norm_layer=norm_layer,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=config.mask_roi_size,
        sampling_ratio=2,
    )
    mask_head = MaskRCNNHeads(
        backbone.out_channels,
        [256, 256, 256, 256],
        1,
        norm_layer=norm_layer,
    )
    return MaskRCNN(
        backbone,
        num_classes=config.num_classes,
        min_size=size,
        max_size=size,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25],
        rpn_anchor_generator=anchors,
        rpn_head=rpn_head,
        rpn_pre_nms_top_n_train=config.rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=config.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=config.rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=config.rpn_post_nms_top_n_test,
        rpn_nms_thresh=config.rpn_nms_thresh,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_score_thresh=0.05,
        box_nms_thresh=config.box_nms_thresh,
        box_detections_per_img=config.max_detections,
        mask_roi_pool=mask_roi_pool,
        mask_head=mask_head,
    )


def build_model(image_size: int | None = None, architecture: str | None = None):
    size = int(image_size or config.image_size)
    architecture = architecture or config.architecture
    if architecture == "v1":
        return _build_v1(size)
    if architecture == "v2":
        return _build_v2(size)
    raise ValueError(f"Unknown Mask R-CNN architecture: {architecture}")
