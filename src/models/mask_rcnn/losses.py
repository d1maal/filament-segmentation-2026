import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads
from torchvision.models.detection.roi_heads import project_masks_on_boxes

from .config import config


def maskrcnn_bce_dice_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    logits = mask_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]
    bce = F.binary_cross_entropy_with_logits(logits, mask_targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * mask_targets).sum(dim=(1, 2))
    denominator = probs.sum(dim=(1, 2)) + mask_targets.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + float(config.mask_dice_weight) * dice.mean()


def install_mask_loss():
    roi_heads.maskrcnn_loss = maskrcnn_bce_dice_loss
