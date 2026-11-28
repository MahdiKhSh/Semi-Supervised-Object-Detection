"""CaRe-SSOD / CoRe-SSOD Core Modules: Photometric Risk, Bidirectional NWD/JS Probing,
and Conformal Soft-Negative Tri-Zone Unsupervised Loss.

Designed to solve the key failure modes of Mean Teacher / Soft Teacher on road
scenes (BDD100K night/rain recall collapse, small-object IoU rejection, RPN/ROI
false-negative background poisoning, and false-positive calibration drift):

1. **Zero-Metadata Multi-Cue Photometric Degradation Risk d(x) in [0, 1]**:
   Estimates per-image visibility degradation directly from RGB pixels via
   luminance darkness, contrast deficit, dark-background specular glare, and
   dark-channel atmospheric haze — zero external weather/time metadata required.
2. **Direct Bidirectional Cross-View RoI Probing + Weighted Box Fusion**:
   Detects on the weak original view (F1) and runs ONLY the teacher backbone on
   the horizontally flipped view (F2 — no redundant second RPN/NMS, no O(N^2)
   matching cliff). Directly probes F2 at flipped box coordinates to obtain:
   - Weighted bidirectional box fusion b_bar (sharper localization, higher IoU),
   - Bidirectional probability consensus p_bar = 0.5*(p1 + p2),
   - Vectorized O(N) scale-adaptive NWD+IoU localization uncertainty u_loc,
   - Vectorized O(N) cross-view Jensen-Shannon divergence u_cls,
   - Mild calibrated reliability q = s_bar * exp(-alpha_loc * u_loc - alpha_cls * u_cls),
   - Cached teacher FPN features F1 reused at zero extra cost for ROI soft-negative
     background weighting.
3. **Conformal Tri-Zone Unsupervised Loss (Single Student FPN Pass)**:
   - **RPN Zone-2 Ambiguity Ignore Masking**: Anchors overlapping Zone-2
     ambiguous objects (and not Zone-1 positives) are marked ignore (-1) so the
     RPN does not crush missed/adverse-weather objects as hard background.
   - **ROI Conformal Soft-Negative Background Weighting**: Background ROI
     proposals are weighted by teacher background probability r_j = p_T(bg | b_j)
     attenuated by (1 - pi_hat_r(q_z)) when overlapping a Zone-2 box (where
     pi_hat_r is the conformal PAVA true-positive probability), normalized to
     mean 1.0 so background calibration is strictly preserved (eliminating
     false-positive explosion).
   - **Task-Decoupled Foreground Supervision**: Zone-1 positives drive
     reliability-weighted classification (with mild full-distribution KL
     distillation) and localization-gated regression (smooth-L1 + sampled-proposal
     GIoU gated by q_loc = 1 - u_loc).
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
    """Label-free multi-cue RGB visibility degradation score d(x) in [0, 1].

    Combines four physical photometric cues without external weather/time labels:
    - ``darkness``: low ITU-R BT.601 luminance (night, dusk, tunnels),
    - ``low_contrast``: compressed luminance standard deviation (fog, haze, overcast),
    - ``glare``: specular bright spots against dark backgrounds (headlights, wet road),
    - ``haze``: elevated dark-channel intensity with washed-out chroma (fog, heavy snow/rain).

    Args:
        image: (3, H, W) float tensor in [0, 1].
    Returns:
        dict with ``degradation``, ``darkness``, ``low_contrast``, ``glare``, ``haze``, ``regime``.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) RGB tensor, got {tuple(image.shape)}")
    img = image.detach().float().clamp(0.0, 1.0)
    # ITU-R BT.601 luminance
    luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    mean_y = float(luma.mean().item())
    std_y = float(luma.std(unbiased=False).item())
    bright_frac = float((luma > 0.92).float().mean().item())

    # Dark-channel + low-chroma atmospheric veil proxy (He et al., CVPR 2009)
    min_c = img.min(dim=0).values
    max_c = img.max(dim=0).values
    dark_chan = float(min_c.mean().item())
    mean_chroma = float((max_c - min_c).mean().item())
    # Fog/heavy mist exhibits high dark channel (>0.35) AND low color saturation (<0.12)
    haze = max(0.0, min(1.0, ((dark_chan - 0.30) / 0.35) * ((0.16 - mean_chroma) / 0.16)))

    darkness = max(0.0, min(1.0, (0.38 - mean_y) / 0.38))
    low_contrast = max(0.0, min(1.0, (0.20 - std_y) / 0.20))
    glare = max(0.0, min(1.0, 4.0 * bright_frac * (1.0 - mean_y)))
    degradation = max(
        0.0,
        min(1.0, 0.48 * darkness + 0.24 * low_contrast + 0.14 * glare + 0.14 * haze),
    )
    return {
        "degradation": round(degradation, 4),
        "darkness": round(darkness, 4),
        "low_contrast": round(low_contrast, 4),
        "glare": round(glare, 4),
        "haze": round(haze, 4),
        "regime": "degraded" if degradation >= 0.28 else "clear",
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


def elementwise_nwd(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    *,
    base_constant: float = 12.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Vectorized O(N) paired scale-adaptive NWD similarity in (0, 1]."""
    if len(boxes_a) == 0:
        return boxes_a.new_zeros((0,))
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
        (cxa - cxb).square()
        + (cya - cyb).square()
        + 0.25 * ((wa - wb).square() + (ha - hb).square())
    )
    w2 = torch.sqrt(w2_sq + eps)
    size_a = torch.sqrt(wa * ha)
    size_b = torch.sqrt(wb * hb)
    c_scale = torch.clamp(0.5 * torch.sqrt(size_a * size_b), min=float(base_constant))
    return torch.exp(-w2 / c_scale)


def elementwise_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Vectorized O(N) paired IoU in [0, 1]."""
    if len(boxes_a) == 0:
        return boxes_a.new_zeros((0,))
    a = boxes_a.float()
    b = boxes_b.float()
    ix1 = torch.maximum(a[:, 0], b[:, 0])
    iy1 = torch.maximum(a[:, 1], b[:, 1])
    ix2 = torch.minimum(a[:, 2], b[:, 2])
    iy2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
    return (inter / (area_a + area_b - inter).clamp(min=eps)).clamp(0.0, 1.0)


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


# ------------------------------------------------------------------ 3. Online calibrator (CaRe-SSOD ablation)
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
        low_ratio: float = 0.65,
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

    def precision_for(self, reliability: float, regime: str = "clear") -> float:
        """Smooth heuristic TP probability fallback when conformal PAVA is unavailable."""
        return max(0.0, min(1.0, float(reliability)))

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


# ------------------------------------------------------------------ 4. Bidirectional Cross-View Probing
def _flip_boxes_x(boxes: torch.Tensor, width: int) -> torch.Tensor:
    out = boxes.clone()
    if len(out):
        old_x1 = out[:, 0].clone()
        old_x2 = out[:, 2].clone()
        out[:, 0] = width - old_x2
        out[:, 2] = width - old_x1
    return out


@torch.no_grad()
def _teacher_detect_with_probs(teacher, images: list[torch.Tensor]):
    """Run teacher detection + extract full (C+1)-class softmax probabilities per box."""
    teacher.eval()
    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
    transformed, _ = teacher.transform(images, None)
    features = teacher.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    proposals, _ = teacher.rpn(transformed, features, None)
    detections, _ = teacher.roi_heads(features, proposals, transformed.image_sizes, None)
    detections = teacher.transform.postprocess(detections, transformed.image_sizes, original_sizes)

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
def teacher_bidirectional_probe(
    teacher,
    images: list[torch.Tensor],
    *,
    use_nwd: bool = True,
    nwd_weight: float = 0.65,
    nwd_constant: float = 12.8,
    loc_alpha: float = 0.35,
    cls_alpha: float = 0.20,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Direct Bidirectional Cross-View Teacher Probing with Weighted Box Fusion.

    1. Runs full teacher detection on original weak view -> FPN F1 and boxes b1.
    2. Runs ONLY teacher backbone on horizontally flipped view -> FPN F2
       (eliminates redundant second RPN + NMS and O(N^2) greedy matching!).
    3. Probes F1 at b1 and F2 at flip(b1) via RoIAlign + box_head + box_predictor
       to obtain paired cross-view probabilities (p1, p2) and refined box b2.
    4. Computes weighted consensus box fusion b_bar, bidirectional score s_bar,
       vectorized O(N) NWD+IoU localization uncertainty u_loc, JS divergence u_cls,
       and calibrated reliability q = s_bar * exp(-loc_alpha*u_loc - cls_alpha*u_cls).
    5. Returns cached F1 features for zero-cost ROI soft-negative background weighting.
    """
    teacher.eval()
    orig_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
    t1, _ = teacher.transform(images, None)
    f1 = teacher.backbone(t1.tensors)
    if isinstance(f1, torch.Tensor):
        f1 = {"0": f1}
    props1, _ = teacher.rpn(t1, f1, None)
    dets1, _ = teacher.roi_heads(f1, props1, t1.image_sizes, None)
    dets1 = teacher.transform.postprocess(dets1, t1.image_sizes, orig_sizes)

    # Flipped view: backbone ONLY (no second RPN, no second NMS)
    flipped_imgs = [torch.flip(im, dims=(-1,)) for im in images]
    t2, _ = teacher.transform(flipped_imgs, None)
    f2 = teacher.backbone(t2.tensors)
    if isinstance(f2, torch.Tensor):
        f2 = {"0": f2}

    boxes_t1: list[torch.Tensor] = []
    boxes_t2: list[torch.Tensor] = []
    counts: list[int] = []
    for det, (oh, ow), (th1, tw1), (th2, tw2) in zip(dets1, orig_sizes, t1.image_sizes, t2.image_sizes):
        b = det["boxes"]
        counts.append(len(b))
        if len(b):
            b1 = b.clone()
            b1[:, [0, 2]] *= float(tw1) / max(1, ow)
            b1[:, [1, 3]] *= float(th1) / max(1, oh)
            boxes_t1.append(b1)

            bf = _flip_boxes_x(b, ow)
            bf[:, [0, 2]] *= float(tw2) / max(1, ow)
            bf[:, [1, 3]] *= float(th2) / max(1, oh)
            boxes_t2.append(bf)
        else:
            boxes_t1.append(b.new_zeros((0, 4)))
            boxes_t2.append(b.new_zeros((0, 4)))

    num_cls = teacher.roi_heads.box_predictor.cls_score.out_features
    if sum(counts) > 0:
        rf1 = teacher.roi_heads.box_roi_pool(f1, boxes_t1, t1.image_sizes)
        rf1 = teacher.roi_heads.box_head(rf1)
        logits1, _ = teacher.roi_heads.box_predictor(rf1)
        probs1_all = F.softmax(logits1.float(), dim=-1)

        rf2 = teacher.roi_heads.box_roi_pool(f2, boxes_t2, t2.image_sizes)
        rf2 = teacher.roi_heads.box_head(rf2)
        logits2, reg2 = teacher.roi_heads.box_predictor(rf2)
        probs2_all = F.softmax(logits2.float(), dim=-1)
        dec2_all = teacher.roi_heads.box_coder.decode(reg2.float(), boxes_t2)

        split_p1 = list(probs1_all.split(counts, dim=0))
        split_p2 = list(probs2_all.split(counts, dim=0))
        split_d2 = list(dec2_all.split(counts, dim=0))
    else:
        split_p1 = [d["boxes"].new_zeros((0, num_cls)) for d in dets1]
        split_p2 = [d["boxes"].new_zeros((0, num_cls)) for d in dets1]
        split_d2 = [d["boxes"].new_zeros((0, num_cls, 4)) for d in dets1]

    probed: list[dict[str, torch.Tensor]] = []
    for det, p1, p2, d2, (oh, ow), (th2, tw2) in zip(
        dets1, split_p1, split_p2, split_d2, orig_sizes, t2.image_sizes
    ):
        b1 = det["boxes"].float()
        labels = det["labels"].long()
        s1 = det["scores"].float()
        if len(b1):
            rows = torch.arange(len(labels), device=labels.device)
            b2 = d2[rows, labels].clone()
            b2[:, [0, 2]] = (b2[:, [0, 2]] * (float(ow) / max(1, tw2))).clamp(0.0, float(ow))
            b2[:, [1, 3]] = (b2[:, [1, 3]] * (float(oh) / max(1, th2))).clamp(0.0, float(oh))
            b2 = _flip_boxes_x(b2, ow)
            # Ensure well-ordered non-degenerate fused boxes
            b2[:, 2] = torch.maximum(b2[:, 2], b2[:, 0] + 1.0)
            b2[:, 3] = torch.maximum(b2[:, 3], b2[:, 1] + 1.0)

            s2_rep = p2[rows, labels].clamp(0.0, 1.0)
            iou_vec = elementwise_iou(b1, b2)
            if use_nwd:
                nwd_vec = elementwise_nwd(b1, b2, base_constant=nwd_constant)
                sim_vec = nwd_weight * nwd_vec + (1.0 - nwd_weight) * iou_vec
            else:
                sim_vec = iou_vec
            u_loc = (1.0 - sim_vec).clamp(0.0, 1.0)
            u_cls = jensen_shannon_divergence(p1, p2)

            # Weighted Bidirectional Box Fusion (empirically +1.06 IoU & +57 TPs)
            w1 = s1.clamp(min=1e-4)[:, None]
            w2 = s2_rep.clamp(min=1e-4)[:, None]
            fused_boxes = (w1 * b1 + w2 * b2) / (w1 + w2)
            fused_boxes[:, 0] = fused_boxes[:, 0].clamp(0.0, float(ow))
            fused_boxes[:, 1] = fused_boxes[:, 1].clamp(0.0, float(oh))
            fused_boxes[:, 2] = torch.maximum(fused_boxes[:, 2].clamp(0.0, float(ow)), fused_boxes[:, 0] + 1.0)
            fused_boxes[:, 3] = torch.maximum(fused_boxes[:, 3].clamp(0.0, float(oh)), fused_boxes[:, 1] + 1.0)

            consensus_score = 0.5 * (s1 + s2_rep)
            reliability = (consensus_score * torch.exp(-float(loc_alpha) * u_loc - float(cls_alpha) * u_cls)).clamp(0.0, 1.0)
            probs_avg = (0.5 * (p1 + p2)).clamp(min=1e-7)
            probs_avg = probs_avg / probs_avg.sum(dim=-1, keepdim=True)
        else:
            fused_boxes = b1.clone()
            consensus_score = s1.clone()
            reliability = s1.clone()
            u_loc = s1.clone()
            u_cls = s1.clone()
            probs_avg = p1.clone()

        probed.append({
            "boxes": fused_boxes.detach(),
            "raw_boxes": b1.detach(),
            "labels": labels.detach(),
            "scores": consensus_score.detach(),
            "raw_scores": s1.detach(),
            "probs": probs_avg.detach(),
            "reliability": reliability.detach(),
            "u_loc": u_loc.detach(),
            "u_cls": u_cls.detach(),
            "q_loc": (1.0 - u_loc).clamp(0.0, 1.0).detach(),
        })

    teacher_cache = {
        "features": {k: v.detach() for k, v in f1.items()},
        "transformed_sizes": list(t1.image_sizes),
        "original_sizes": orig_sizes,
    }
    return probed, teacher_cache


@torch.no_grad()
def extract_teacher_tri_zone_targets(
    teacher,
    images: list[torch.Tensor],
    calibrator: Any,
    *,
    use_nwd: bool = True,
    use_visibility_calibration: bool = True,
    use_tri_zone: bool = True,
    nwd_weight: float = 0.65,
    nwd_constant: float = 12.8,
    loc_alpha: float = 0.35,
    cls_alpha: float = 0.20,
    return_cache: bool = False,
):
    """Generate Tri-Zone pseudo-targets via Bidirectional Cross-View Probing."""
    probed, teacher_cache = teacher_bidirectional_probe(
        teacher,
        images,
        use_nwd=use_nwd,
        nwd_weight=nwd_weight,
        nwd_constant=nwd_constant,
        loc_alpha=loc_alpha,
        cls_alpha=cls_alpha,
    )

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

    for image, item in zip(images, probed):
        vis = estimate_visibility_degradation(image)
        regime = str(vis["regime"])
        deg_scores.append(float(vis["degradation"]))
        if hasattr(calibrator, "observe_unlabeled_visibility"):
            calibrator.observe_unlabeled_visibility(float(vis["degradation"]), regime)

        boxes = item["boxes"]
        labels = item["labels"]
        scores = item["scores"]
        probs = item["probs"]
        reliability = item["reliability"]
        u_loc = item["u_loc"]
        u_cls = item["u_cls"]
        q_loc = item["q_loc"]
        num_cls = probs.shape[-1]

        cal_labels, cal_scores = [], []
        z1_idx, z2_idx = [], []
        z1_ptp, z2_ptp = [], []
        labels_list = labels.detach().cpu().tolist()
        scores_list = scores.detach().cpu().tolist()
        rel_list = reliability.detach().cpu().tolist()

        has_pava = hasattr(calibrator, "precision_for")
        eff_regime = regime if use_visibility_calibration else "clear"

        for j, (cls_id, sc, q) in enumerate(zip(labels_list, scores_list, rel_list)):
            if not (math.isfinite(sc) and math.isfinite(q)):
                continue
            if sc >= 0.25:
                cal_labels.append(int(cls_id))
                cal_scores.append(float(sc))
            tau_hi, tau_lo = calibrator.thresholds(cls_id, regime, use_visibility=use_visibility_calibration)
            p_tp = float(calibrator.precision_for(q, eff_regime)) if has_pava else float(q)
            if q >= tau_hi:
                z1_idx.append(j)
                z1_ptp.append(p_tp)
            elif use_tri_zone and q >= tau_lo:
                z2_idx.append(j)
                z2_ptp.append(p_tp)

        if cal_scores and hasattr(calibrator, "update"):
            calibrator.update(cal_labels, cal_scores, regime)

        def _slice(indices: list[int], ptp_vals: list[float]):
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
                    "q_loc": q_loc[sel].detach(),
                    "p_tp": torch.tensor(ptp_vals, dtype=torch.float32, device=boxes.device),
                }
            return {
                "boxes": boxes.new_zeros((0, 4)),
                "labels": labels.new_zeros((0,), dtype=torch.long),
                "scores": scores.new_zeros((0,)),
                "probs": probs.new_zeros((0, num_cls)),
                "reliability": scores.new_zeros((0,)),
                "u_loc": scores.new_zeros((0,)),
                "u_cls": scores.new_zeros((0,)),
                "q_loc": scores.new_zeros((0,)),
                "p_tp": scores.new_zeros((0,)),
            }

        z1 = _slice(z1_idx, z1_ptp)
        z2 = _slice(z2_idx, z2_ptp)
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
    if return_cache:
        return outputs, stats, teacher_cache
    return outputs, stats


# ------------------------------------------------------------------ 5. Strong asymmetric view
def strong_view_care(
    image: torch.Tensor,
    zone1: dict[str, torch.Tensor],
    zone2: dict[str, torch.Tensor],
    *,
    rng,
    enable_cutout: bool = True,
    return_flip: bool = False,
):
    """Asymmetric strong student augmentation synchronizing Zone-1 and Zone-2 boxes."""
    img = image.clone()
    img = TF.adjust_brightness(img, rng.uniform(0.75, 1.25))
    img = TF.adjust_contrast(img, rng.uniform(0.75, 1.25))
    img = TF.adjust_saturation(img, rng.uniform(0.75, 1.25))
    if rng.random() < 0.10:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    img = img.clamp(0.0, 1.0)

    z1 = {k: v.clone() for k, v in zone1.items()}
    z2 = {k: v.clone() for k, v in zone2.items()}

    flipped = rng.random() < 0.5
    if flipped:
        img = TF.hflip(img)
        w = int(img.shape[-1])
        z1["boxes"] = _flip_boxes_x(z1["boxes"], w)
        z2["boxes"] = _flip_boxes_x(z2["boxes"], w)

    # Box-safe mild Random Erasing / Cutout (1-2 small patches simulating occlusion/glare)
    if enable_cutout and rng.random() < 0.40:
        _, h, w = img.shape
        n_patches = rng.randint(1, 2)
        for _ in range(n_patches):
            ph = max(4, int(h * rng.uniform(0.03, 0.10)))
            pw = max(4, int(w * rng.uniform(0.03, 0.10)))
            y0 = rng.randint(0, max(0, h - ph))
            x0 = rng.randint(0, max(0, w - pw))
            fill_val = rng.uniform(0.0, 1.0)
            img[:, y0:y0 + ph, x0:x0 + pw] = fill_val

    if return_flip:
        return img, z1, z2, flipped
    return img, z1, z2


# ------------------------------------------------------------------ 6. Ambiguity masking & Conformal Soft-Negative Loss
def filter_ambiguous_background_proposals(
    proposals: list[torch.Tensor],
    pos_boxes_list: list[torch.Tensor],
    ambig_boxes_list: list[torch.Tensor],
    *,
    pos_iou_thresh: float = 0.5,
    ambig_iou_thresh: float = 0.3,
) -> tuple[list[torch.Tensor], int]:
    """Legacy helper: exclude non-positive proposals overlapping Zone-2 boxes."""
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
        keep = is_pos | (~is_ambig)
        if int(keep.sum().item()) == 0:
            filtered.append(props[:1])
        else:
            suppressed_total += int((~keep).sum().item())
            filtered.append(props[keep])
    return filtered, suppressed_total


def _rpn_forward_with_zone2_ignore(
    rpn,
    images,
    features: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    zone2_boxes_list: list[torch.Tensor],
    *,
    ignore_iou_thresh: float = 0.30,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], int]:
    """Run TorchVision RPN (preserving module hooks) while ignoring negative anchors overlapping Zone-2 boxes."""
    orig_assign = rpn.assign_targets_to_anchors
    ignored_anchors = [0]

    def _assign_with_zone2_ignore(anchors, tgts):
        labels, matched_gt_boxes = orig_assign(anchors, tgts)
        for img_idx, (anc, lbl, z2_b) in enumerate(zip(anchors, labels, zone2_boxes_list)):
            if len(z2_b) == 0 or len(anc) == 0:
                continue
            neg_inds = torch.where(lbl == 0)[0]
            if len(neg_inds) == 0:
                continue
            max_z2_iou = box_iou(anc[neg_inds], z2_b).max(dim=1).values
            ambig_neg = neg_inds[max_z2_iou >= ignore_iou_thresh]
            if len(ambig_neg) > 0:
                lbl = lbl.clone()
                lbl[ambig_neg] = -1.0  # -1 is ignored by BalancedPositiveNegativeSampler
                ignored_anchors[0] += int(len(ambig_neg))
                labels[img_idx] = lbl
        return labels, matched_gt_boxes

    rpn.assign_targets_to_anchors = _assign_with_zone2_ignore
    try:
        boxes, rpn_losses = rpn(images, features, targets)
    finally:
        rpn.assign_targets_to_anchors = orig_assign
    return boxes, rpn_losses, ignored_anchors[0]


def _kl_soft_distillation(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Temperature-calibrated KL divergence weighted by reliability (no T^2 blowup)."""
    if len(student_logits) == 0:
        return student_logits.sum() * 0.0
    t = max(0.5, float(temperature))
    tp = teacher_probs.detach().float().clamp(min=eps)
    tp_log_t = torch.log(tp) / t
    tp_t = F.softmax(tp_log_t, dim=-1)
    log_ps_t = F.log_softmax(student_logits.float() / t, dim=-1)
    # Linear T scaling keeps gradient magnitude matched to CE without blowing up stability loss
    kl_per_box = F.kl_div(log_ps_t, tp_t, reduction="none").sum(dim=-1) * t
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
    teacher=None,
    teacher_cache: dict[str, Any] | None = None,
    flips: list[bool] | None = None,
    use_tri_zone: bool = True,
    giou_weight: float = 0.25,
    kl_zone1_weight: float = 0.15,
    kl_zone2_weight: float = 0.05,
    temperature: float = 1.5,
    reg_loc_thresh: float = 0.55,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Single-pass feature-shared CoRe-SSOD / CaRe-SSOD unsupervised loss.

    Computes student FPN features ONCE on ``aug_images`` and applies:
    1. **RPN with Zone-2 Ambiguity Ignore Masking**: prevents crushing borderline
       Zone-2 objects as hard RPN negatives while keeping clean background supervision.
    2. **ROI Conformal Soft-Negative Background Weighting**: weights background ROI
       proposals by teacher background probability ``p_T(bg)`` attenuated by
       ``(1 - p_tp(z2))`` for Zone-2 overlaps, normalized to mean 1.0 (preserves
       background calibration and eliminates false-positive spam!).
    3. **Task-Decoupled Foreground Supervision**: reliability-weighted ROI CE +
       localization-gated Smooth-L1 & sampled-proposal GIoU regression (only on
       geometrically stable Zone-1 boxes with ``q_loc >= reg_loc_thresh``).
    4. **Calibrated Full-Distribution KL Distillation**: mild soft distillation on
       Zone-1 and Zone-2 boxes using bidirectional consensus probabilities (including
       background mass).
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

    # 1. Zone-1 RPN + Conformal Soft-Negative ROI loss ONLY on images with >= 1 Zone-1 box
    #    (An unlabeled image with 0 Zone-1 pseudo-boxes is unknown foreground — e.g. dark night —
    #     NOT a verified empty negative image! Full feature maps are retained below for Zone-2 KL.)
    if has_z1 and transformed_targets is not None:
        positive = [i for i, target in enumerate(transformed_targets) if len(target["boxes"]) > 0]
        if len(positive) == len(aug_images):
            det_images = transformed
            det_features = features
            det_targets = transformed_targets
            det_orig_sizes = original_sizes
            det_z1_meta = zone1_meta
            det_z2_meta = zone2_meta
            det_ambig = ambig_trans
            det_flips = flips
            det_t_cache = teacher_cache
        else:
            from torchvision.models.detection.image_list import ImageList

            det_images = ImageList(
                transformed.tensors[positive],
                [transformed.image_sizes[i] for i in positive],
            )
            det_features = {k: v[positive] for k, v in features.items()}
            det_targets = [transformed_targets[i] for i in positive]
            det_orig_sizes = [original_sizes[i] for i in positive]
            det_z1_meta = [zone1_meta[i] for i in positive]
            det_z2_meta = [zone2_meta[i] for i in positive]
            det_ambig = [ambig_trans[i] for i in positive]
            det_flips = [flips[i] for i in positive] if flips is not None else None
            det_t_cache = (
                {
                    "features": {k: v[positive] for k, v in teacher_cache["features"].items()},
                    "transformed_sizes": [teacher_cache["transformed_sizes"][i] for i in positive],
                }
                if teacher_cache is not None
                else None
            )

        det_has_z2 = use_tri_zone and any(len(b) > 0 for b in det_ambig)
        if det_has_z2:
            proposals, rpn_losses, ignored_rpn = _rpn_forward_with_zone2_ignore(
                model.rpn, det_images, det_features, det_targets, det_ambig
            )
            suppressed_bg += ignored_rpn
        else:
            proposals, rpn_losses = model.rpn(det_images, det_features, det_targets)

        sampled_props, matched_idxs, labels_list, reg_targets_list = model.roi_heads.select_training_samples(
            proposals, det_targets
        )
        box_features = model.roi_heads.box_roi_pool(det_features, sampled_props, det_images.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        labels = torch.cat(labels_list, dim=0)
        regression_targets = torch.cat(reg_targets_list, dim=0)
        fg_mask = labels > 0
        bg_mask = labels == 0

        # Compute Conformal Soft-Negative weights for background ROI proposals
        cls_weights = torch.ones(len(labels), dtype=torch.float32, device=labels.device)

        # (a) Foreground reliability weights from matched Zone-1 boxes
        fg_rel_per_prop = []
        fg_qloc_per_prop = []
        for m_idx, z1_m in zip(matched_idxs, det_z1_meta):
            rel = z1_m.get("reliability")
            qloc = z1_m.get("q_loc")
            if rel is None or len(rel) == 0:
                fg_rel_per_prop.append(torch.ones(len(m_idx), dtype=torch.float32, device=m_idx.device))
                fg_qloc_per_prop.append(torch.ones(len(m_idx), dtype=torch.float32, device=m_idx.device))
            else:
                clamped = m_idx.clamp(min=0, max=len(rel) - 1)
                fg_rel_per_prop.append(rel.to(m_idx.device)[clamped].float())
                if qloc is not None and len(qloc) == len(rel):
                    fg_qloc_per_prop.append(qloc.to(m_idx.device)[clamped].float())
                else:
                    fg_qloc_per_prop.append(torch.ones(len(m_idx), dtype=torch.float32, device=m_idx.device))
        fg_rel_all = torch.cat(fg_rel_per_prop, dim=0)
        fg_qloc_all = torch.cat(fg_qloc_per_prop, dim=0)
        if bool(fg_mask.any()):
            # Smooth reliability weight in [0.5, 1.0] for Zone-1 positives
            cls_weights[fg_mask] = (0.5 + 0.5 * fg_rel_all[fg_mask].clamp(0.0, 1.0)).detach()

        # (b) Background soft weights: teacher p_T(bg) * (1 - p_tp(z2 overlap)), normalized to mean 1.0
        if bool(bg_mask.any()):
            # Base teacher p_T(bg) via cached F1 features (zero extra backbone pass!)
            t_bg_all = None
            if teacher is not None and det_t_cache is not None and det_flips is not None:
                t_props_list = []
                for props, was_flipped, s_sz, o_sz, t_sz in zip(
                    sampled_props,
                    det_flips,
                    det_images.image_sizes,
                    det_orig_sizes,
                    det_t_cache["transformed_sizes"],
                ):
                    p = props.detach().clone()
                    p[:, [0, 2]] *= float(o_sz[1]) / max(1, s_sz[1])
                    p[:, [1, 3]] *= float(o_sz[0]) / max(1, s_sz[0])
                    if was_flipped:
                        p = _flip_boxes_x(p, o_sz[1])
                    p[:, [0, 2]] *= float(t_sz[1]) / max(1, o_sz[1])
                    p[:, [1, 3]] *= float(t_sz[0]) / max(1, o_sz[0])
                    t_props_list.append(p)
                with torch.no_grad():
                    t_roi_f = teacher.roi_heads.box_roi_pool(
                        det_t_cache["features"], t_props_list, det_t_cache["transformed_sizes"]
                    )
                    t_roi_f = teacher.roi_heads.box_head(t_roi_f)
                    t_logits, _ = teacher.roi_heads.box_predictor(t_roi_f)
                    t_bg_all = F.softmax(t_logits.float(), dim=-1)[:, 0].detach()

            # Conformal Zone-2 attenuation: (1 - pi_hat_r(q_z2)) when proposal overlaps a Zone-2 box
            z2_atten_parts = []
            for props, z2_b, z2_m in zip(sampled_props, det_ambig, det_z2_meta):
                atten = torch.ones(len(props), dtype=torch.float32, device=props.device)
                if use_tri_zone and len(z2_b) > 0 and len(props) > 0:
                    iou_z2 = box_iou(props.detach(), z2_b)
                    max_iou, best_z2 = iou_z2.max(dim=1)
                    ov_mask = max_iou >= 0.30
                    if bool(ov_mask.any()):
                        p_tp = z2_m.get("p_tp")
                        if p_tp is None or len(p_tp) != len(z2_b):
                            p_tp = z2_m.get("reliability", torch.full((len(z2_b),), 0.5, device=props.device))
                        p_tp_matched = p_tp.to(props.device)[best_z2[ov_mask]].float().clamp(0.0, 0.90)
                        atten[ov_mask] = 1.0 - p_tp_matched
                        suppressed_bg += int(ov_mask.sum().item())
                z2_atten_parts.append(atten)
            z2_atten_all = torch.cat(z2_atten_parts, dim=0)

            bg_w = z2_atten_all[bg_mask]
            if t_bg_all is not None:
                bg_w = bg_w * t_bg_all[bg_mask].clamp(min=0.05)
            bg_w = bg_w.clamp(min=0.05)
            # Normalize background weights to mean 1.0 -> NEVER breaks background calibration!
            cls_weights[bg_mask] = (bg_w / bg_w.mean().clamp(min=1e-4)).detach()

        per_sample_cls = F.cross_entropy(class_logits.float(), labels, reduction="none")
        loss_cls = (per_sample_cls * cls_weights).sum() / cls_weights.sum().clamp(min=1.0)

        # Localization-gated Smooth-L1 + Sampled-Proposal GIoU regression on Zone-1 foregrounds
        reg_gate = fg_mask & (fg_qloc_all >= float(reg_loc_thresh))
        sampled_pos_inds = torch.where(reg_gate)[0]
        if len(sampled_pos_inds) > 0:
            n_total, num_classes = class_logits.shape
            box_reg = box_regression.reshape(n_total, num_classes, 4)
            labels_pos = labels[sampled_pos_inds]
            pred_deltas = box_reg[sampled_pos_inds, labels_pos].float()
            tgt_deltas = regression_targets[sampled_pos_inds].float()
            q_w = fg_qloc_all[sampled_pos_inds].detach().clamp(0.1, 1.0)
            per_box_l1 = F.smooth_l1_loss(pred_deltas, tgt_deltas, beta=1.0 / 9.0, reduction="none").sum(dim=1)
            loss_box = (per_box_l1 * q_w).sum() / max(1.0, float(labels.numel()))

            # Sampled-proposal GIoU refinement (operates on actual noisy RPN proposals, not zero-jitter GT!)
            if giou_weight > 0.0:
                all_props_cat = torch.cat(sampled_props, dim=0)
                gt_matched_parts = []
                for m_idx, tgt_t in zip(matched_idxs, det_targets):
                    gb = tgt_t["boxes"].detach()
                    if len(gb) == 0:
                        gt_matched_parts.append(all_props_cat.new_zeros((len(m_idx), 4)))
                    else:
                        gt_matched_parts.append(gb[m_idx.clamp(min=0, max=len(gb) - 1)])
                gt_matched_all = torch.cat(gt_matched_parts, dim=0)
                pos_props = all_props_cat[sampled_pos_inds].float()
                pos_gt = gt_matched_all[sampled_pos_inds].float()
                dec_pos = model.roi_heads.box_coder.decode_single(pred_deltas, pos_props)
                loss_giou = float(giou_weight) * _giou_consistency_loss(dec_pos, pos_gt, q_w)
            else:
                loss_giou = box_regression.sum() * 0.0
        else:
            loss_box = box_regression.sum() * 0.0
            loss_giou = box_regression.sum() * 0.0

        det_total = sum(rpn_losses.values()) + loss_cls + loss_box
        total = det_total + loss_giou
        det_loss_val = float(det_total.detach())
        stab_loss_val += float(loss_giou.detach())

    # 2. Mild calibrated full-distribution KL distillation on Zone 1 (reusing student FPN)
    z1_boxes_trans = _scale_boxes_to_transformed(
        [z["boxes"] for z in zone1_meta], original_sizes, transformed.image_sizes
    )
    if kl_zone1_weight > 0.0 and sum(len(b) for b in z1_boxes_trans) > 0:
        roi_f1 = model.roi_heads.box_roi_pool(features, z1_boxes_trans, transformed.image_sizes)
        roi_f1 = model.roi_heads.box_head(roi_f1)
        logits1, _ = model.roi_heads.box_predictor(roi_f1)
        probs1 = torch.cat([z["probs"] for z in zone1_meta], dim=0)
        rel1 = torch.cat([z["reliability"] for z in zone1_meta], dim=0)
        loss_kl1 = float(kl_zone1_weight) * _kl_soft_distillation(logits1, probs1, rel1, temperature=temperature)
        total = loss_kl1 if total is None else (total + loss_kl1)
        stab_loss_val += float(loss_kl1.detach())

    # 3. Mild calibrated full-distribution KL distillation on Zone 2 (includes background mass)
    if use_tri_zone and kl_zone2_weight > 0.0 and sum(len(b) for b in ambig_trans) > 0:
        roi_f2 = model.roi_heads.box_roi_pool(features, ambig_trans, transformed.image_sizes)
        roi_f2 = model.roi_heads.box_head(roi_f2)
        logits2, _ = model.roi_heads.box_predictor(roi_f2)
        probs2 = torch.cat([z["probs"] for z in zone2_meta], dim=0)
        p_tp2 = torch.cat(
            [z.get("p_tp", z["reliability"]) for z in zone2_meta], dim=0
        )
        loss_kl2 = float(kl_zone2_weight) * _kl_soft_distillation(logits2, probs2, p_tp2, temperature=temperature)
        total = loss_kl2 if total is None else (total + loss_kl2)
        stab_loss_val += float(loss_kl2.detach())

    return total, {
        "unsup": det_loss_val,
        "stability": stab_loss_val,
        "suppressed_bg": suppressed_bg,
    }
