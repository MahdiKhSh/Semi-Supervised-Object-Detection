"""Differentiable, matched-object multi-level consistency prototype.

Student boxes and logits MUST correspond to teacher pseudo boxes after
homography projection. Teacher targets are detached (no teacher gradients).
This alone does not constitute a new published method.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_consistency_loss(
    class_logits, pred_boxes, pseudo_boxes, pseudo_labels, pseudo_scores,
    localization_uncertainty, *, box_weight=1.0, class_weight=1.0,
    confidence_weight=0.25, reliability_scale=4.0, eps=1e-6
):
    """Return independent differentiable scalar losses.

    class_logits: (N,C+1) with background in the last column.
    pred_boxes and pseudo_boxes: (N,4) xyxy in *same augmented image space*.
    labels: (N,) integers in [0,C); scores and uncertainty: (N,).
    Local box deltas are normalized by pseudo box width and height.
    """
    if class_logits.ndim != 2 or pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
        raise ValueError("invalid student prediction shapes")
    n, class_dim = class_logits.shape
    if class_dim < 2 or pred_boxes.shape[0] != n:
        raise ValueError("batch and class dimensions must agree")
    labels = pseudo_labels.long().detach().reshape(-1)
    scores = pseudo_scores.detach().reshape(-1).float()
    uncertainty = localization_uncertainty.detach().reshape(-1).float()
    boxes = pseudo_boxes.detach().reshape(-1, 4)
    if any(t.shape[0] != n for t in (labels, scores, uncertainty, boxes)):
        raise ValueError("pseudo target lengths mismatch")
    if n == 0:
        zero = class_logits.sum() * 0.0 + pred_boxes.sum() * 0.0
        return {"loss_stability_box": zero, "loss_stability_class": zero,
                "loss_stability_confidence": zero}
    if bool(((labels < 0) | (labels >= class_dim-1)).any()):
        raise ValueError("invalid foreground pseudo label")
    if reliability_scale < 0 or any(v < 0 for v in (box_weight, class_weight, confidence_weight)):
        raise ValueError("loss weights must be nonnegative")
    valid = torch.isfinite(scores) & torch.isfinite(uncertainty) & (uncertainty >= 0)
    valid &= torch.isfinite(boxes).all(dim=1) & torch.isfinite(pred_boxes).all(dim=1)
    valid &= torch.isfinite(class_logits).all(dim=1)
    wh = boxes[:, 2:] - boxes[:, :2]
    valid &= (wh > eps).all(dim=1) & (scores >= 0) & (scores <= 1)
    # Invalid examples cannot contribute and must not inject nan into autograd.
    mask = valid.float()
    reliable = (scores.clamp(0, 1) * torch.exp(-reliability_scale * uncertainty.clamp(min=0)))
    weight = torch.where(valid, reliable, torch.zeros_like(reliable)).detach()
    denom = weight.sum().clamp(min=eps)
    normalized_box_delta = (torch.where(valid[:, None], pred_boxes, boxes) - boxes).abs()
    normalized_box_delta /= wh.clamp(min=eps).repeat(1, 2)
    spatial = F.smooth_l1_loss(normalized_box_delta, torch.zeros_like(normalized_box_delta), reduction="none").mean(dim=1)
    safe_logits = torch.where(valid[:, None], class_logits, torch.zeros_like(class_logits))
    cls = F.cross_entropy(safe_logits.float(), labels, reduction="none")
    probability = F.softmax(safe_logits.float(), dim=1).gather(1, labels[:, None]).flatten()
    conf = (probability - scores).square()
    return {
        "loss_stability_box": box_weight * (spatial * weight).sum() / denom,
        "loss_stability_class": class_weight * (cls * weight).sum() / denom,
        "loss_stability_confidence": confidence_weight * (conf * weight).sum() / denom,
    }
