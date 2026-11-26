"""CaRe-SSOD: Self-Calibrating Photometric-Risk & Tri-Zone NWD Soft-Distillation.

A unified, zero-metadata semi-supervised road object detection method designed
specifically to address the failure modes of Soft Teacher / Mean Teacher on
BDD100K (night/rain recall collapse, small-object localization rejection,
false-negative background poisoning, and double-backbone overhead):

1. **Zero-Metadata Photometric Visibility Estimation**:
   Estimates per-image visibility degradation d(x) in [0, 1] directly from RGB
   luminance, local contrast, and dark-background glare without requiring
   external weather or time-of-day labels on unlabeled images.
2. **Scale-Aware NWD + Cross-View JS Reliability**:
   Measures cross-view localization stability via scale-adaptive Normalized
   Gaussian Wasserstein Distance (preventing unfair rejection of small traffic
   lights/signs/pedestrians) coupled with Jensen-Shannon divergence over the
   teacher's full (C+1)-class probability distribution.
3. **Online Dual-Axis (Class x Visibility) EMA Threshold Calibration**:
   Dynamically adapts high/low zone thresholds per class and per visibility
   regime from unlabeled teacher statistics (checkpoint-serializable).
4. **Tri-Zone Partitioning + Single-Pass Ambiguity Background Suppression**:
   - Zone 1 (High Reliability): Positive RPN/ROI targets + reliability-weighted
     GIoU box consistency + temperature-scaled KL soft distillation.
   - Zone 2 (Ambiguous): Shielded from hard background classification loss
     (proposals overlapping Zone-2 boxes are excluded from false-negative
     background sampling) and trained solely via reliability-weighted KL
     soft distillation — even on images with zero Zone-1 detections.
   - Zone 3 (Low Reliability): Background.
   Reuses a single student FPN forward pass per unlabeled batch.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou, generalized_box_iou


# ------------------------------------------------------------------ 1. Visibility estimator
@torch.no_grad()
def estimate_visibility_degradation(image: torch.Tensor) -> dict[str, float | str]:
    """Label-free RGB visibility degradation score d(x) in [0, 1].

    Args:
        image: (3, H, W) float tensor in [0, 1].
    Returns:
        dict with ``degradation``, ``darkness``, ``low_contrast``, ``glare``, ``regime``.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) RGB tensor, got {tuple(image.shape)}")
    img = image.detach().float().clamp(0.0, 1.0)
    # ITU-R BT.601 luminance
    luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    mean_y = float(luma.mean().item())
    std_y = float(luma.std(unbiased=False).item())
    bright_frac = float((luma > 0.92).float().mean().item())

    darkness = max(0.0, min(1.0, (0.35 - mean_y) / 0.35))
    low_contrast = max(0.0, min(1.0, (0.18 - std_y) / 0.18))
    glare = max(0.0, min(1.0, 4.0 * bright_frac * (1.0 - mean_y)))
    degradation = max(0.0, min(1.0, 0.55 * darkness + 0.30 * low_contrast + 0.15 * glare))
    return {
        "degradation": round(degradation, 4),
        "darkness": round(darkness, 4),
        "low_contrast": round(low_contrast, 4),
        "glare": round(glare, 4),
        "regime": "degraded" if degradation >= 0.35 else "clear",
    }


# ------------------------------------------------------------------ 2. NWD & JS stability
def pairwise_nwd(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    *,
    base_constant: float = 12.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-adaptive Normalized Gaussian Wasserstein Distance similarity in (0, 1].

    Models each xyxy box as a 2D Gaussian N((cx, cy), diag(w^2/4, h^2/4)).
    Adapts the normalization constant C = max(base_constant, 0.5 * sqrt(sqrt(area_a * area_b)))
    so tiny objects (traffic lights/signs) are not unfairly penalized by 1-2px
    shifts while large objects retain tight spatial sensitivity.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return boxes_a.new_zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a.float()
    b = boxes_b.float()
    wa = (a[:, 2] - a[:, 0]).clamp(min=eps)
    ha = (a[:, 3] - a[:, 1]).clamp(min=eps)
    cxa = a[:, 0] + 0.5 * wa
    cya = a[:, 1] + 0.5 * ha

    wb = (b[:, 2] - b[:, 0]).clamp(min=eps)
    hb = (b[:, 3] - b[:, 1]).clamp(min=eps)
    cxb = b[:, 0] + 0.5 * wb
    cyb = b[:, 1] + 0.5 * hb

    w2_sq = (
        (cxa[:, None] - cxb[None, :]).square()
        + (cya[:, None] - cyb[None, :]).square()
        + 0.25 * ((wa[:, None] - wb[None, :]).square() + (ha[:, None] - hb[None, :]).square())
    )
    w2 = torch.sqrt(w2_sq + eps)
    size_a = torch.sqrt(wa * ha)
    size_b = torch.sqrt(wb * hb)
    c_scale = torch.clamp(0.5 * torch.sqrt(size_a[:, None] * size_b[None, :]), min=float(base_constant))
    return torch.exp(-w2 / c_scale)


def jensen_shannon_divergence(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Normalized Jensen-Shannon divergence in [0, 1] between two probability rows."""
    pa = probs_a.float().clamp(min=eps)
    pa = pa / pa.sum(dim=-1, keepdim=True)
    pb = probs_b.float().clamp(min=eps)
    pb = pb / pb.sum(dim=-1, keepdim=True)
    m = 0.5 * (pa + pb)
    kl_a = (pa * (pa.log() - m.log())).sum(dim=-1)
    kl_b = (pb * (pb.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)
    return js.clamp(0.0, 1.0)


# ------------------------------------------------------------------ 3. Online calibrator
class OnlineVisibilityClassCalibrator:
    """Checkpoint-serializable dual-axis (class x visibility) EMA threshold calibrator."""

    def __init__(
        self,
        num_foreground_classes: int = 10,
        *,
        base_threshold: float = 0.7,
        momentum: float = 0.99,
        class_gamma: float = 0.25,
        visibility_gamma: float = 0.30,
        min_high: float = 0.42,
        max_high: float = 0.88,
        min_low: float = 0.18,
        low_ratio: float = 0.50,
    ):
        self.num_classes = int(num_foreground_classes)
        self.base_threshold = float(base_threshold)
        self.momentum = float(momentum)
        self.class_gamma = float(class_gamma)
        self.visibility_gamma = float(visibility_gamma)
        self.min_high = float(min_high)
        self.max_high = float(max_high)
        self.min_low = float(min_low)
        self.low_ratio = float(low_ratio)
        # Foreground class indices are 1..num_classes (TorchVision convention).
        self.class_ema: dict[int, float] = {c: float(base_threshold) for c in range(1, self.num_classes + 1)}
        self.visibility_ema: dict[str, float] = {"clear": float(base_threshold), "degraded": float(base_threshold)}
        self.updates: int = 0

    def update(self, labels: list[int], scores: list[float], regime: str) -> None:
        """Update online EMA confidence statistics from teacher detections."""
        if not scores:
            return
        m = self.momentum
        mean_score = sum(scores) / len(scores)
        key = "degraded" if regime == "degraded" else "clear"
        self.visibility_ema[key] = m * self.visibility_ema[key] + (1.0 - m) * float(mean_score)

        by_cls: dict[int, list[float]] = {}
        for cls_id, sc in zip(labels, scores):
            if cls_id in self.class_ema:
                by_cls.setdefault(cls_id, []).append(float(sc))
        for cls_id, vals in by_cls.items():
            cls_mean = sum(vals) / len(vals)
            self.class_ema[cls_id] = m * self.class_ema[cls_id] + (1.0 - m) * cls_mean
        self.updates += 1

    def thresholds(self, cls_id: int, regime: str, *, use_visibility: bool = True) -> tuple[float, float]:
        """Return (tau_high, tau_low) for foreground class ``cls_id`` under ``regime``."""
        max_cls = max(self.class_ema.values(), default=self.base_threshold)
        cls_mu = self.class_ema.get(int(cls_id), self.base_threshold)
        cls_ratio = max(0.25, min(1.0, cls_mu / max(1e-4, max_cls)))
        cls_factor = cls_ratio ** self.class_gamma

        vis_factor = 1.0
        if use_visibility and regime == "degraded":
            clear_mu = max(1e-4, self.visibility_ema.get("clear", self.base_threshold))
            deg_mu = self.visibility_ema.get("degraded", clear_mu)
            vis_ratio = max(0.40, min(1.15, deg_mu / clear_mu))
            vis_factor = vis_ratio ** self.visibility_gamma

        tau_high = max(self.min_high, min(self.max_high, self.base_threshold * cls_factor * vis_factor))
        tau_low = max(self.min_low, min(tau_high - 0.05, tau_high * self.low_ratio))
        return float(tau_high), float(tau_low)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "base_threshold": self.base_threshold,
            "momentum": self.momentum,
            "class_gamma": self.class_gamma,
            "visibility_gamma": self.visibility_gamma,
            "min_high": self.min_high,
            "max_high": self.max_high,
            "min_low": self.min_low,
            "low_ratio": self.low_ratio,
            "class_ema": {int(k): float(v) for k, v in self.class_ema.items()},
            "visibility_ema": {str(k): float(v) for k, v in self.visibility_ema.items()},
            "updates": int(self.updates),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "class_ema" in state:
            self.class_ema = {int(k): float(v) for k, v in state["class_ema"].items()}
        if "visibility_ema" in state:
            self.visibility_ema = {str(k): float(v) for k, v in state["visibility_ema"].items()}
        self.updates = int(state.get("updates", 0))


# ------------------------------------------------------------------ 4. Teacher full-dist extraction
@torch.no_grad()
def _teacher_detect_with_probs(teacher, images: list[torch.Tensor]):
    """Run teacher detection + extract full (C+1)-class softmax probabilities per box.

    Reuses the teacher's own FPN features (zero redundant backbone pass).
    """
    teacher.eval()
    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
    transformed, _ = teacher.transform(images, None)
    features = teacher.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    proposals, _ = teacher.rpn(transformed, features, None)
    detections, _ = teacher.roi_heads(features, proposals, transformed.image_sizes, None)
    detections = teacher.transform.postprocess(detections, transformed.image_sizes, original_sizes)

    # Extract full (C+1)-class probabilities at the kept postprocessed boxes in transformed coords.
    transformed_boxes = []
    for det, orig_sz, trans_sz in zip(detections, original_sizes, transformed.image_sizes):
        boxes = det["boxes"]
        if len(boxes):
            sy = float(trans_sz[0]) / max(1, orig_sz[0])
            sx = float(trans_sz[1]) / max(1, orig_sz[1])
            scaled = boxes.clone()
            scaled[:, [0, 2]] *= sx
            scaled[:, [1, 3]] *= sy
            transformed_boxes.append(scaled)
        else:
            transformed_boxes.append(boxes.new_zeros((0, 4)))

    counts = [len(b) for b in transformed_boxes]
    if sum(counts) > 0:
        roi_feats = teacher.roi_heads.box_roi_pool(features, transformed_boxes, transformed.image_sizes)
        roi_feats = teacher.roi_heads.box_head(roi_feats)
        cls_logits, _ = teacher.roi_heads.box_predictor(roi_feats)
        all_probs = F.softmax(cls_logits.float(), dim=-1)
        split_probs = list(all_probs.split(counts, dim=0))
    else:
        num_cls = teacher.roi_heads.box_predictor.cls_score.out_features
        split_probs = [det["boxes"].new_zeros((0, num_cls)) for det in detections]

    for det, probs in zip(detections, split_probs):
        det["probs"] = probs.detach()
    return detections


@torch.no_grad()
def extract_teacher_tri_zone_targets(
    teacher,
    images: list[torch.Tensor],
    calibrator: OnlineVisibilityClassCalibrator,
    *,
    use_nwd: bool = True,
    use_visibility_calibration: bool = True,
    use_tri_zone: bool = True,
    nwd_weight: float = 0.65,
    nwd_constant: float = 12.8,
    loc_alpha: float = 2.5,
    cls_alpha: float = 1.5,
    match_floor: float = 0.40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate Tri-Zone pseudo-targets with scale-aware NWD + JS reliability."""
    standard = _teacher_detect_with_probs(teacher, images)
    flipped_imgs = [torch.flip(im, dims=(-1,)) for im in images]
    repeated = _teacher_detect_with_probs(teacher, flipped_imgs)

    outputs: list[dict[str, Any]] = []
    stats = {
        "candidates": 0,
        "accepted": 0,
        "zone1_count": 0,
        "zone2_count": 0,
        "quality_sum": 0.0,
        "degradation_mean": 0.0,
    }
    deg_scores = []

    for i, (image, pred, rep) in enumerate(zip(images, standard, repeated)):
        vis = estimate_visibility_degradation(image)
        regime = str(vis["regime"])
        deg_scores.append(float(vis["degradation"]))
        if hasattr(calibrator, "observe_unlabeled_visibility"):
            calibrator.observe_unlabeled_visibility(float(vis["degradation"]), regime)

        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        probs = pred["probs"]
        num_cls = probs.shape[-1]

        # Flip repeated boxes back to original image coordinates
        b_boxes = rep["boxes"].clone()
        width = image.shape[-1]
        if len(b_boxes):
            old_x1 = b_boxes[:, 0].clone()
            old_x2 = b_boxes[:, 2].clone()
            b_boxes[:, 0] = width - old_x2
            b_boxes[:, 2] = width - old_x1

        u_loc = torch.ones_like(scores)
        u_cls = torch.ones_like(scores)

        if len(boxes) and len(b_boxes):
            iou_mat = box_iou(boxes, b_boxes)
            if use_nwd:
                nwd_mat = pairwise_nwd(boxes, b_boxes, base_constant=nwd_constant)
                sim_mat = nwd_weight * nwd_mat + (1.0 - nwd_weight) * iou_mat
            else:
                sim_mat = iou_mat
            sim_mat[labels[:, None] != rep["labels"][None, :]] = -1.0
            sim_cpu = sim_mat.detach().cpu().numpy()
            matches = sorted(
                ((float(sim_cpu[a, b]), a, b)
                 for a in range(len(boxes))
                 for b in range(len(b_boxes))
                 if sim_cpu[a, b] >= match_floor),
                reverse=True,
            )
            seen_a, seen_b = set(), set()
            for sim_val, a, b in matches:
                if a in seen_a or b in seen_b:
                    continue
                seen_a.add(a)
                seen_b.add(b)
                u_loc[a] = max(0.0, min(1.0, 1.0 - sim_val))
                u_cls[a] = jensen_shannon_divergence(probs[a:a + 1], rep["probs"][b:b + 1])[0]

        reliability = scores * torch.exp(-loc_alpha * u_loc - cls_alpha * u_cls)

        # Update online calibrator with non-trivial teacher detections (score >= 0.25)
        cal_labels, cal_scores = [], []
        z1_idx, z2_idx = [], []
        labels_list = labels.detach().cpu().tolist()
        scores_list = scores.detach().cpu().tolist()
        rel_list = reliability.detach().cpu().tolist()

        for j, (cls_id, sc, q) in enumerate(zip(labels_list, scores_list, rel_list)):
            if not (math.isfinite(sc) and math.isfinite(q)):
                continue
            if sc >= 0.25:
                cal_labels.append(int(cls_id))
                cal_scores.append(float(sc))
            tau_hi, tau_lo = calibrator.thresholds(cls_id, regime, use_visibility=use_visibility_calibration)
            if q >= tau_hi and sc >= tau_lo:
                z1_idx.append(j)
            elif use_tri_zone and (q >= tau_lo or sc >= tau_hi):
                z2_idx.append(j)

        if cal_scores:
            calibrator.update(cal_labels, cal_scores, regime)

        def _slice(indices: list[int]):
            if indices:
                sel = torch.tensor(indices, dtype=torch.long, device=boxes.device)
                return {
                    "boxes": boxes[sel].detach(),
                    "labels": labels[sel].detach(),
                    "scores": scores[sel].detach(),
                    "probs": probs[sel].detach(),
                    "reliability": reliability[sel].detach(),
                    "u_loc": u_loc[sel].detach(),
                    "u_cls": u_cls[sel].detach(),
                }
            return {
                "boxes": boxes.new_zeros((0, 4)),
                "labels": labels.new_zeros((0,), dtype=torch.long),
                "scores": scores.new_zeros((0,)),
                "probs": probs.new_zeros((0, num_cls)),
                "reliability": scores.new_zeros((0,)),
                "u_loc": scores.new_zeros((0,)),
                "u_cls": scores.new_zeros((0,)),
            }

        z1 = _slice(z1_idx)
        z2 = _slice(z2_idx)
        stats["candidates"] += len(scores)
        stats["accepted"] += len(z1_idx)
        stats["zone1_count"] += len(z1_idx)
        stats["zone2_count"] += len(z2_idx)
        if z1_idx:
            stats["quality_sum"] += float(z1["reliability"].sum().item())

        outputs.append({
            "zone1": z1,
            "zone2": z2,
            "visibility": vis,
        })

    stats["degradation_mean"] = round(sum(deg_scores) / max(1, len(deg_scores)), 4)
    return outputs, stats


# ------------------------------------------------------------------ 5. Strong asymmetric view
def _flip_boxes_x(boxes: torch.Tensor, width: int) -> torch.Tensor:
    out = boxes.clone()
    if len(out):
        old_x1 = out[:, 0].clone()
        old_x2 = out[:, 2].clone()
        out[:, 0] = width - old_x2
        out[:, 2] = width - old_x1
    return out


def strong_view_care(
    image: torch.Tensor,
    zone1: dict[str, torch.Tensor],
    zone2: dict[str, torch.Tensor],
    *,
    rng,
    enable_cutout: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Asymmetric strong student augmentation synchronizing Zone-1 and Zone-2 boxes."""
    img = image.clone()
    img = TF.adjust_brightness(img, rng.uniform(0.65, 1.35))
    img = TF.adjust_contrast(img, rng.uniform(0.65, 1.35))
    img = TF.adjust_saturation(img, rng.uniform(0.65, 1.35))
    if rng.random() < 0.15:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    img = img.clamp(0.0, 1.0)

    z1 = {k: v.clone() for k, v in zone1.items()}
    z2 = {k: v.clone() for k, v in zone2.items()}

    if rng.random() < 0.5:
        img = TF.hflip(img)
        w = int(img.shape[-1])
        z1["boxes"] = _flip_boxes_x(z1["boxes"], w)
        z2["boxes"] = _flip_boxes_x(z2["boxes"], w)

    # Box-safe Random Erasing / Cutout (1-2 small patches simulating occlusion/glare)
    if enable_cutout and rng.random() < 0.5:
        _, h, w = img.shape
        n_patches = rng.randint(1, 2)
        for _ in range(n_patches):
            ph = max(4, int(h * rng.uniform(0.04, 0.14)))
            pw = max(4, int(w * rng.uniform(0.04, 0.14)))
            y0 = rng.randint(0, max(0, h - ph))
            x0 = rng.randint(0, max(0, w - pw))
            fill_val = rng.uniform(0.0, 1.0)
            img[:, y0:y0 + ph, x0:x0 + pw] = fill_val

    return img, z1, z2


# ------------------------------------------------------------------ 6. Ambiguity suppression & loss
def filter_ambiguous_background_proposals(
    proposals: list[torch.Tensor],
    pos_boxes_list: list[torch.Tensor],
    ambig_boxes_list: list[torch.Tensor],
    *,
    pos_iou_thresh: float = 0.5,
    ambig_iou_thresh: float = 0.3,
) -> tuple[list[torch.Tensor], int]:
    """Exclude non-positive proposals overlapping Zone-2 ambiguous boxes.

    Eliminates false-negative background gradient poisoning without mutating
    TorchVision's internal FasterRCNN loss functions.
    """
    filtered: list[torch.Tensor] = []
    suppressed_total = 0
    for props, pos_b, amb_b in zip(proposals, pos_boxes_list, ambig_boxes_list):
        if len(props) == 0 or len(amb_b) == 0:
            filtered.append(props)
            continue
        amb_iou = box_iou(props, amb_b).max(dim=1).values
        is_ambig = amb_iou >= ambig_iou_thresh
        if len(pos_b) > 0:
            pos_iou = box_iou(props, pos_b).max(dim=1).values
            is_pos = pos_iou >= pos_iou_thresh
        else:
            is_pos = torch.zeros_like(is_ambig)
        # Keep positive proposals and clean background; drop ambiguous background
        keep = is_pos | (~is_ambig)
        if int(keep.sum().item()) == 0:
            # Never pass an empty proposal tensor to roi_heads
            filtered.append(props[:1])
        else:
            suppressed_total += int((~keep).sum().item())
            filtered.append(props[keep])
    return filtered, suppressed_total


def _kl_soft_distillation(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Temperature-calibrated KL divergence weighted by reliability q_i."""
    if len(student_logits) == 0:
        return student_logits.sum() * 0.0
    t = max(0.5, float(temperature))
    # Re-sharpen/soften teacher probability distribution cleanly at temperature T
    tp = teacher_probs.detach().float().clamp(min=eps)
    tp_log_t = torch.log(tp) / t
    tp_t = F.softmax(tp_log_t, dim=-1)
    log_ps_t = F.log_softmax(student_logits.float() / t, dim=-1)
    kl_per_box = F.kl_div(log_ps_t, tp_t, reduction="none").sum(dim=-1) * (t * t)
    w = weights.detach().float().clamp(min=0.0)
    return (kl_per_box * w).sum() / w.sum().clamp(min=eps)


def _giou_consistency_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant reliability-weighted Generalized IoU loss."""
    if len(pred_boxes) == 0:
        return pred_boxes.sum() * 0.0
    # Ensure well-ordered xyxy boxes for GIoU numerical stability
    px1 = torch.minimum(pred_boxes[:, 0], pred_boxes[:, 2])
    py1 = torch.minimum(pred_boxes[:, 1], pred_boxes[:, 3])
    px2 = torch.maximum(pred_boxes[:, 0], pred_boxes[:, 2]) + eps
    py2 = torch.maximum(pred_boxes[:, 1], pred_boxes[:, 3]) + eps
    safe_pred = torch.stack([px1, py1, px2, py2], dim=1)

    tx1 = torch.minimum(target_boxes[:, 0], target_boxes[:, 2])
    ty1 = torch.minimum(target_boxes[:, 1], target_boxes[:, 3])
    tx2 = torch.maximum(target_boxes[:, 0], target_boxes[:, 2]) + eps
    ty2 = torch.maximum(target_boxes[:, 1], target_boxes[:, 3]) + eps
    safe_tgt = torch.stack([tx1, ty1, tx2, ty2], dim=1).detach()

    # Diagonal entries of generalized_box_iou give paired GIoU in [-1, 1]
    giou = torch.diag(generalized_box_iou(safe_pred.float(), safe_tgt.float())).clamp(-1.0, 1.0)
    loss_per_box = 1.0 - giou
    w = weights.detach().float().clamp(min=0.0)
    return (loss_per_box * w).sum() / w.sum().clamp(min=eps)


def _scale_boxes_to_transformed(
    boxes_list: list[torch.Tensor],
    original_sizes: list[tuple[int, int]],
    transformed_sizes: list[tuple[int, int]],
) -> list[torch.Tensor]:
    scaled_list = []
    for boxes, (oh, ow), (th, tw) in zip(boxes_list, original_sizes, transformed_sizes):
        if len(boxes):
            out = boxes.clone()
            out[:, [0, 2]] *= float(tw) / max(1, ow)
            out[:, [1, 3]] *= float(th) / max(1, oh)
            scaled_list.append(out.detach())
        else:
            scaled_list.append(boxes.new_zeros((0, 4)))
    return scaled_list


def care_unsupervised_loss(
    model,
    aug_images: list[torch.Tensor],
    zone1_targets: list[dict[str, torch.Tensor]],
    zone1_meta: list[dict[str, torch.Tensor]],
    zone2_meta: list[dict[str, torch.Tensor]],
    *,
    use_tri_zone: bool = True,
    giou_weight: float = 0.5,
    kl_zone1_weight: float = 0.5,
    kl_zone2_weight: float = 0.35,
    temperature: float = 2.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Single-pass feature-shared CaRe-SSOD unsupervised loss.

    Computes student FPN features ONCE on ``aug_images`` and uses them for:
    1. RPN + Ambiguity-Suppressed ROI detection loss on Zone-1 positives.
    2. Reliability-weighted GIoU + KL soft distillation on Zone-1 boxes.
    3. Reliability-weighted KL soft distillation on Zone-2 ambiguous boxes
       (even when an image has zero Zone-1 detections).
    """
    has_z1 = any(len(t["boxes"]) > 0 for t in zone1_targets)
    has_z2 = use_tri_zone and any(len(z["boxes"]) > 0 for z in zone2_meta)
    if not has_z1 and not has_z2:
        return None, {"unsup": 0.0, "stability": 0.0, "suppressed_bg": 0}

    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in aug_images]
    transformed, transformed_targets = model.transform(aug_images, zone1_targets if has_z1 else None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}

    ambig_trans = _scale_boxes_to_transformed(
        [z["boxes"] for z in zone2_meta], original_sizes, transformed.image_sizes
    )

    total = None
    det_loss_val = 0.0
    stab_loss_val = 0.0
    suppressed_bg = 0

    # 1. Zone-1 supervised-style detection loss with Zone-2 background suppression
    if has_z1 and transformed_targets is not None:
        # A missing pseudo-label is unknown foreground, not a negative image.
        # Retain the full feature maps below for Zone-2-only distillation.
        positive = [i for i, target in enumerate(transformed_targets) if len(target["boxes"])]
        det_images, det_features, det_targets = transformed, features, transformed_targets
        det_ambiguous = ambig_trans
        if len(positive) != len(aug_images):
            from torchvision.models.detection.image_list import ImageList

            det_images = ImageList(transformed.tensors[positive], [transformed.image_sizes[i] for i in positive])
            det_features = {k: v[positive] for k, v in features.items()}
            det_targets = [transformed_targets[i] for i in positive]
            det_ambiguous = [ambig_trans[i] for i in positive]
        proposals, rpn_losses = model.rpn(det_images, det_features, det_targets)
        if use_tri_zone and has_z2:
            pos_trans = [t["boxes"].detach() for t in det_targets]
            proposals, suppressed_bg = filter_ambiguous_background_proposals(
                proposals, pos_trans, det_ambiguous
            )
        _, roi_losses = model.roi_heads(det_features, proposals, det_images.image_sizes, det_targets)
        det_total = sum(rpn_losses.values()) + sum(roi_losses.values())
        total = det_total
        det_loss_val = float(det_total.detach())

    # 2. Single-pass feature-shared ROI distillation & GIoU on Zone 1 and Zone 2
    z1_boxes_trans = _scale_boxes_to_transformed(
        [z["boxes"] for z in zone1_meta], original_sizes, transformed.image_sizes
    )
    if sum(len(b) for b in z1_boxes_trans) > 0:
        roi_f1 = model.roi_heads.box_roi_pool(features, z1_boxes_trans, transformed.image_sizes)
        roi_f1 = model.roi_heads.box_head(roi_f1)
        logits1, reg1 = model.roi_heads.box_predictor(roi_f1)
        decoded1 = model.roi_heads.box_coder.decode(reg1, z1_boxes_trans)
        labels1 = torch.cat([z["labels"] for z in zone1_meta], dim=0).long()
        probs1 = torch.cat([z["probs"] for z in zone1_meta], dim=0)
        rel1 = torch.cat([z["reliability"] for z in zone1_meta], dim=0)
        gt1 = torch.cat(z1_boxes_trans, dim=0)
        rows1 = torch.arange(len(labels1), device=labels1.device)
        chosen1 = decoded1[rows1, labels1]

        loss_giou1 = giou_weight * _giou_consistency_loss(chosen1, gt1, rel1)
        loss_kl1 = kl_zone1_weight * _kl_soft_distillation(logits1, probs1, rel1, temperature=temperature)
        z1_aux = loss_giou1 + loss_kl1
        total = z1_aux if total is None else (total + z1_aux)
        stab_loss_val += float(z1_aux.detach())

    if use_tri_zone and sum(len(b) for b in ambig_trans) > 0:
        roi_f2 = model.roi_heads.box_roi_pool(features, ambig_trans, transformed.image_sizes)
        roi_f2 = model.roi_heads.box_head(roi_f2)
        logits2, _ = model.roi_heads.box_predictor(roi_f2)
        probs2 = torch.cat([z["probs"] for z in zone2_meta], dim=0)
        rel2 = torch.cat([z["reliability"] for z in zone2_meta], dim=0)
        loss_kl2 = kl_zone2_weight * _kl_soft_distillation(logits2, probs2, rel2, temperature=temperature)
        total = loss_kl2 if total is None else (total + loss_kl2)
        stab_loss_val += float(loss_kl2.detach())

    return total, {
        "unsup": det_loss_val,
        "stability": stab_loss_val,
        "suppressed_bg": suppressed_bg,
    }
