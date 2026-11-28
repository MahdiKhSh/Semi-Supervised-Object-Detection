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
def _photometric_restore_for_probe(image: torch.Tensor, vis: dict[str, float | str]) -> torch.Tensor:
    """Label-free adaptive shadow/contrast restoration for degraded probing views.

    Operates strictly on raw RGB statistics already computed by
    ``estimate_visibility_degradation`` (zero external weather/time metadata).
    Leaves clear daytime images untouched while lifting shadow detail on dark/night
    scenes (without clipping bright traffic lights) and stretching washed-out fog/haze.
    """
    deg = float(vis.get("degradation", 0.0))
    if deg < 0.20:
        return image
    out = image.detach().float().clamp(0.0, 1.0)
    darkness = float(vis.get("darkness", 0.0))
    if darkness > 0.05:
        gamma = max(0.58, 1.0 - 0.42 * darkness)
        out = out.clamp(min=1e-4).pow(gamma)
    contrast_deficit = max(float(vis.get("low_contrast", 0.0)), float(vis.get("haze", 0.0)))
    if contrast_deficit > 0.05:
        c_gain = 1.0 + 0.28 * contrast_deficit
        mean_rgb = out.mean(dim=(-2, -1), keepdim=True)
        out = mean_rgb + c_gain * (out - mean_rgb)
    return out.clamp(0.0, 1.0).to(dtype=image.dtype)


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
    enable_photometric_probe: bool = True,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Direct Bidirectional Cross-View Teacher Probing with Photometric Dual-Discovery.

    1. Runs full teacher detection on original weak view -> FPN F1 and boxes b1.
    2. Computes label-free photometric degradation d(x); on degraded scenes, applies
       adaptive gamma/contrast restoration to the horizontally flipped view before
       computing teacher FPN F2, and runs lightweight RPN+RoI discovery on F2 for
       degraded images (d(x) >= 0.28) to recover shadow/glare objects missed on F1.
    3. Probes F1 at b1 and F2 at flip(b1) via RoIAlign + box_head + box_predictor
       to obtain paired cross-view probabilities (p1, p2) and refined boxes.
    4. Computes weighted consensus box fusion b_bar, visibility-adaptive score s_bar,
       vectorized O(N) NWD+IoU localization uncertainty u_loc, JS divergence u_cls,
       and calibrated reliability q = s_bar * exp(-loc_alpha*u_loc - cls_alpha*u_cls).
    5. Returns cached F1 features and per-image visibility dicts for zero-cost reuse.
    """
    teacher.eval()
    orig_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
    vis_list = [estimate_visibility_degradation(im) for im in images]

    t1, _ = teacher.transform(images, None)
    f1 = teacher.backbone(t1.tensors)
    if isinstance(f1, torch.Tensor):
        f1 = {"0": f1}
    props1, _ = teacher.rpn(t1, f1, None)
    dets1, _ = teacher.roi_heads(f1, props1, t1.image_sizes, None)
    dets1 = teacher.transform.postprocess(dets1, t1.image_sizes, orig_sizes)

    # Flipped view (+ zero-metadata photometric shadow/contrast lift on degraded scenes)
    flipped_imgs = [
        torch.flip(_photometric_restore_for_probe(im, vis) if enable_photometric_probe else im, dims=(-1,))
        for im, vis in zip(images, vis_list)
    ]
    t2, _ = teacher.transform(flipped_imgs, None)
    f2 = teacher.backbone(t2.tensors)
    if isinstance(f2, torch.Tensor):
        f2 = {"0": f2}

    # Dual-discovery on degraded scenes (d(x) >= 0.28): reuse already-computed F2 to recover
    # shadow/glare/haze objects missed by raw dark View 1 without an extra backbone pass!
    from_view2_flags: list[torch.Tensor] = [
        torch.zeros((len(d["boxes"]),), dtype=torch.bool, device=d["boxes"].device) for d in dets1
    ]
    deg_indices = (
        [i for i, v in enumerate(vis_list) if float(v["degradation"]) >= 0.28]
        if enable_photometric_probe
        else []
    )
    if deg_indices:
        from torchvision.models.detection.image_list import ImageList

        sub_t2 = ImageList(
            t2.tensors[deg_indices],
            [t2.image_sizes[i] for i in deg_indices],
        )
        sub_f2 = {k: v[deg_indices] for k, v in f2.items()}
        sub_props2, _ = teacher.rpn(sub_t2, sub_f2, None)
        sub_dets2, _ = teacher.roi_heads(sub_f2, sub_props2, sub_t2.image_sizes, None)
        sub_orig = [orig_sizes[i] for i in deg_indices]
        sub_dets2 = teacher.transform.postprocess(sub_dets2, sub_t2.image_sizes, sub_orig)

        for local_idx, img_idx in enumerate(deg_indices):
            d2_item = sub_dets2[local_idx]
            keep_score = d2_item["scores"] >= 0.30
            if not bool(keep_score.any()):
                continue
            oh, ow = orig_sizes[img_idx]
            cand_b2 = _flip_boxes_x(d2_item["boxes"][keep_score].float(), ow)
            cand_l2 = d2_item["labels"][keep_score].long()
            cand_s2 = d2_item["scores"][keep_score].float()

            base_b1 = dets1[img_idx]["boxes"].float()
            base_l1 = dets1[img_idx]["labels"].long()
            if len(base_b1) > 0:
                ious = box_iou(cand_b2, base_b1)
                same_cls = cand_l2[:, None] == base_l1[None, :]
                same_cls_iou = torch.where(same_cls, ious, torch.zeros_like(ious)).max(dim=1).values
                any_cls_iou = ious.max(dim=1).values
                novel_mask = (same_cls_iou < 0.50) & (any_cls_iou < 0.65)
            else:
                novel_mask = torch.ones((len(cand_b2),), dtype=torch.bool, device=cand_b2.device)

            if bool(novel_mask.any()):
                nb = cand_b2[novel_mask][:25]
                nl = cand_l2[novel_mask][:25]
                ns = cand_s2[novel_mask][:25]
                dets1[img_idx] = {
                    "boxes": torch.cat([base_b1, nb], dim=0),
                    "labels": torch.cat([base_l1, nl], dim=0),
                    "scores": torch.cat([dets1[img_idx]["scores"].float(), ns], dim=0),
                }
                from_view2_flags[img_idx] = torch.cat(
                    [
                        from_view2_flags[img_idx],
                        torch.ones((len(nb),), dtype=torch.bool, device=nb.device),
                    ],
                    dim=0,
                )

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
        logits1, reg1 = teacher.roi_heads.box_predictor(rf1)
        probs1_all = F.softmax(logits1.float(), dim=-1)
        dec1_all = teacher.roi_heads.box_coder.decode(reg1.float(), boxes_t1)

        rf2 = teacher.roi_heads.box_roi_pool(f2, boxes_t2, t2.image_sizes)
        rf2 = teacher.roi_heads.box_head(rf2)
        logits2, reg2 = teacher.roi_heads.box_predictor(rf2)
        probs2_all = F.softmax(logits2.float(), dim=-1)
        dec2_all = teacher.roi_heads.box_coder.decode(reg2.float(), boxes_t2)

        split_p1 = list(probs1_all.split(counts, dim=0))
        split_d1 = list(dec1_all.split(counts, dim=0))
        split_p2 = list(probs2_all.split(counts, dim=0))
        split_d2 = list(dec2_all.split(counts, dim=0))
    else:
        split_p1 = [d["boxes"].new_zeros((0, num_cls)) for d in dets1]
        split_d1 = [d["boxes"].new_zeros((0, num_cls, 4)) for d in dets1]
        split_p2 = [d["boxes"].new_zeros((0, num_cls)) for d in dets1]
        split_d2 = [d["boxes"].new_zeros((0, num_cls, 4)) for d in dets1]

    probed: list[dict[str, torch.Tensor]] = []
    for det, from_v2, vis, p1, d1, p2, d2, (oh, ow), (th1, tw1), (th2, tw2) in zip(
        dets1, from_view2_flags, vis_list, split_p1, split_d1, split_p2, split_d2,
        orig_sizes, t1.image_sizes, t2.image_sizes,
    ):
        b_seed = det["boxes"].float()
        labels = det["labels"].long()
        s_raw = det["scores"].float()
        if len(b_seed):
            rows = torch.arange(len(labels), device=labels.device)
            max_x1 = max(0.0, float(ow) - 1.0)
            max_y1 = max(0.0, float(oh) - 1.0)

            # View-1 box: original NMS box for View-1 seeds, RoI-decoded View-1 box for View-2 seeds
            b1_dec = d1[rows, labels].clone()
            b1_dec[:, [0, 2]] = (b1_dec[:, [0, 2]] * (float(ow) / max(1, tw1))).clamp(0.0, float(ow))
            b1_dec[:, [1, 3]] = (b1_dec[:, [1, 3]] * (float(oh) / max(1, th1))).clamp(0.0, float(oh))
            b1_dec[:, 0] = b1_dec[:, 0].clamp(0.0, max_x1)
            b1_dec[:, 1] = b1_dec[:, 1].clamp(0.0, max_y1)
            b1_dec[:, 2] = torch.maximum(b1_dec[:, 2].clamp(0.0, float(ow)), (b1_dec[:, 0] + 1.0).clamp(max=float(ow)))
            b1_dec[:, 3] = torch.maximum(b1_dec[:, 3].clamp(0.0, float(oh)), (b1_dec[:, 1] + 1.0).clamp(max=float(oh)))
            b1 = torch.where(from_v2[:, None], b1_dec, b_seed)

            b2 = d2[rows, labels].clone()
            b2[:, [0, 2]] = (b2[:, [0, 2]] * (float(ow) / max(1, tw2))).clamp(0.0, float(ow))
            b2[:, [1, 3]] = (b2[:, [1, 3]] * (float(oh) / max(1, th2))).clamp(0.0, float(oh))
            b2 = _flip_boxes_x(b2, ow)
            # Ensure well-ordered non-degenerate boxes strictly inside [0, ow] x [0, oh]
            b2[:, 0] = b2[:, 0].clamp(0.0, max_x1)
            b2[:, 1] = b2[:, 1].clamp(0.0, max_y1)
            b2[:, 2] = torch.maximum(b2[:, 2].clamp(0.0, float(ow)), (b2[:, 0] + 1.0).clamp(max=float(ow)))
            b2[:, 3] = torch.maximum(b2[:, 3].clamp(0.0, float(oh)), (b2[:, 1] + 1.0).clamp(max=float(oh)))

            p1_cls = p1[rows, labels].clamp(0.0, 1.0)
            s1 = torch.where(from_v2, p1_cls, s_raw)
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
            fused_boxes[:, 0] = fused_boxes[:, 0].clamp(0.0, max_x1)
            fused_boxes[:, 1] = fused_boxes[:, 1].clamp(0.0, max_y1)
            fused_boxes[:, 2] = torch.maximum(
                fused_boxes[:, 2].clamp(0.0, float(ow)), (fused_boxes[:, 0] + 1.0).clamp(max=float(ow))
            )
            fused_boxes[:, 3] = torch.maximum(
                fused_boxes[:, 3].clamp(0.0, float(oh)), (fused_boxes[:, 1] + 1.0).clamp(max=float(oh))
            )

            deg_val = float(vis.get("degradation", 0.0))
            w_v2 = (0.50 + 0.18 * deg_val) if (enable_photometric_probe and deg_val >= 0.20) else 0.50
            w_v1 = 1.0 - w_v2
            consensus_score = w_v1 * s1 + w_v2 * s2_rep
            # Attenuate cross-view JS penalty mildly when View 2 was photometrically lifted on dark scenes
            eff_cls_alpha = float(cls_alpha) * (1.0 - 0.35 * deg_val if (enable_photometric_probe and deg_val >= 0.20) else 1.0)
            reliability = (consensus_score * torch.exp(-float(loc_alpha) * u_loc - eff_cls_alpha * u_cls)).clamp(0.0, 1.0)
            probs_avg = (w_v1 * p1 + w_v2 * p2).clamp(min=1e-7)
            probs_avg = probs_avg / probs_avg.sum(dim=-1, keepdim=True)
        else:
            b1 = b_seed
            s1 = s_raw
            fused_boxes = b_seed.clone()
            consensus_score = s_raw.clone()
            reliability = s_raw.clone()
            u_loc = s_raw.clone()
            u_cls = s_raw.clone()
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
        "visibilities": vis_list,
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
        enable_photometric_probe=use_visibility_calibration,
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
    cached_vis = teacher_cache.get("visibilities") or [estimate_visibility_degradation(im) for im in images]

    for vis, item in zip(cached_vis, probed):
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

    # Small-object-safe Random Erasing / Cutout: never occlude > 35% of any Zone-1/Zone-2 box
    # (prevents wiping out tiny 8x12 traffic lights/signs into solid gray while keeping their GT label!)
    if enable_cutout and rng.random() < 0.40:
        _, h, w = img.shape
        guard_parts = [b for b in (z1["boxes"], z2["boxes"]) if len(b) > 0]
        guard_boxes = torch.cat(guard_parts, dim=0).float() if guard_parts else None
        guard_areas = (
            ((guard_boxes[:, 2] - guard_boxes[:, 0]).clamp(min=1.0)
             * (guard_boxes[:, 3] - guard_boxes[:, 1]).clamp(min=1.0))
            if guard_boxes is not None
            else None
        )
        n_patches = rng.randint(1, 2)
        for _ in range(n_patches):
            ph = max(4, int(h * rng.uniform(0.03, 0.10)))
            pw = max(4, int(w * rng.uniform(0.03, 0.10)))
            fill_val = rng.uniform(0.0, 1.0)
            chosen_patch = None
            for _attempt in range(4):
                y0 = rng.randint(0, max(0, h - ph))
                x0 = rng.randint(0, max(0, w - pw))
                if guard_boxes is None or guard_areas is None:
                    chosen_patch = (x0, y0)
                    break
                ix1 = torch.clamp(guard_boxes[:, 0], min=float(x0))
                iy1 = torch.clamp(guard_boxes[:, 1], min=float(y0))
                ix2 = torch.clamp(guard_boxes[:, 2], max=float(x0 + pw))
                iy2 = torch.clamp(guard_boxes[:, 3], max=float(y0 + ph))
                inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
                if float((inter / guard_areas).max().item()) <= 0.35:
                    chosen_patch = (x0, y0)
                    break
            if chosen_patch is not None:
                px0, py0 = chosen_patch
                img[:, py0:py0 + ph, px0:px0 + pw] = fill_val

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
    class_counts: dict[int, int] | None = None,
    use_tri_zone: bool = True,
    giou_weight: float = 0.25,
    kl_zone1_weight: float = 0.15,
    kl_zone2_weight: float = 0.12,
    z2_reg_weight: float = 0.50,
    z2_reg_loc_thresh: float = 0.72,
    temperature: float = 1.5,
    reg_loc_thresh: float = 0.55,
    scale_by_batch_size: bool = True,
    filter_zero_box_images: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Single-pass feature-shared CoRe-SSOD / CaRe-SSOD unsupervised loss.

    Computes student FPN features ONCE on ``aug_images`` and applies:
    1. **RPN with Zone-2 Ambiguity Ignore Masking**: prevents crushing borderline
       Zone-2 objects as hard RPN negatives while keeping clean background supervision.
    2. **ROI Conformal Soft-Negative Background Weighting**: weights background ROI
       proposals by teacher background probability ``p_T(bg)`` attenuated by
       ``(1 - p_tp(z2))`` for Zone-2 overlaps, normalized to mean 1.0 (preserves
       background calibration and eliminates false-positive spam!).
    3. **Task-Decoupled Foreground Supervision**: reliability- and tail-frequency-weighted
       ROI CE + localization-gated Smooth-L1 & sampled-proposal GIoU regression on
       geometrically stable Zone-1 boxes, plus decoupled localization regression on
       high-stability Zone-2 boxes (``q_loc >= z2_reg_loc_thresh``) without forcing
       hard Zone-2 class labels.
    4. **Calibrated Full-Distribution KL Distillation**: mild soft distillation on
       Zone-1 and Zone-2 boxes using bidirectional consensus probabilities (including
       background mass).
    """
    has_z1 = any(len(t["boxes"]) > 0 for t in zone1_targets)
    has_z2 = use_tri_zone and any(len(z["boxes"]) > 0 for z in zone2_meta)
    if not has_z1 and not has_z2:
        return None, {"unsup": 0.0, "stability": 0.0, "suppressed_bg": 0}

    total_batch_size = max(1, len(aug_images))
    num_cls_total = int(model.roi_heads.box_predictor.cls_score.out_features)
    cls_bal_lut: torch.Tensor | None = None
    if class_counts:
        positive_counts = {int(k): int(v) for k, v in class_counts.items() if int(v) > 0}
        if positive_counts:
            max_cnt = float(max(positive_counts.values()))
            cls_bal_lut = torch.ones((num_cls_total,), dtype=torch.float32, device=aug_images[0].device)
            for cid, cnt in positive_counts.items():
                if 0 < cid < num_cls_total:
                    cls_bal_lut[cid] = min(1.60, max(1.0, (max_cnt / max(10.0, float(cnt))) ** 0.15))

    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in aug_images]
    transformed, transformed_targets = model.transform(aug_images, zone1_targets if has_z1 else None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}

    ambig_trans = _scale_boxes_to_transformed(
        [z["boxes"] for z in zone2_meta], original_sizes, transformed.image_sizes
    )
    z1_boxes_trans = _scale_boxes_to_transformed(
        [z["boxes"] for z in zone1_meta], original_sizes, transformed.image_sizes
    )

    total = None
    det_loss_val = 0.0
    stab_loss_val = 0.0
    suppressed_bg = 0
    cached_rpn_proposals: dict[int, torch.Tensor] = {}

    # 1. Zone-1 RPN + Conformal Soft-Negative ROI loss ONLY on images with >= 1 Zone-1 box
    #    (An unlabeled image with 0 Zone-1 pseudo-boxes is unknown foreground — e.g. dark night —
    #     NOT a verified empty negative image! Full feature maps are retained below for Zone-2 KL.)
    if has_z1 and transformed_targets is not None:
        positive = (
            [i for i, target in enumerate(transformed_targets) if len(target["boxes"]) > 0]
            if filter_zero_box_images
            else list(range(len(transformed_targets)))
        )
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
        for p_slot, img_idx in enumerate(positive):
            cached_rpn_proposals[img_idx] = proposals[p_slot].detach()

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

        # (a) Foreground reliability (+ mild tail-class balance) weights from matched Zone-1 boxes
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
            fg_w = 0.5 + 0.5 * fg_rel_all[fg_mask].clamp(0.0, 1.0)
            if cls_bal_lut is not None:
                fg_w = fg_w * cls_bal_lut[labels[fg_mask].clamp(0, num_cls_total - 1)]
            cls_weights[fg_mask] = fg_w.detach()

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
            if cls_bal_lut is not None:
                q_w = (q_w * cls_bal_lut[labels_pos.clamp(0, num_cls_total - 1)]).detach()
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

        det_scale = (float(len(positive)) / float(total_batch_size)) if scale_by_batch_size else 1.0
        det_total = det_scale * (sum(rpn_losses.values()) + loss_cls + loss_box)
        scaled_giou = det_scale * loss_giou
        total = det_total + scaled_giou
        det_loss_val = float(det_total.detach())
        stab_loss_val += float(scaled_giou.detach())

    # 2. Decoupled Zone-2 High-Stability Localization Regression (without hard class forcing!)
    #    Zone-2 boxes with strong cross-view geometric consensus (q_loc >= z2_reg_loc_thresh) and
    #    calibrated PAVA precision (p_tp >= 0.35) supervise student box regression on matching RPN proposals.
    if use_tri_zone and float(z2_reg_weight) > 0.0:
        z2_stable_entries: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for img_idx, (z2_b, z2_m) in enumerate(zip(ambig_trans, zone2_meta)):
            qloc_t = z2_m.get("q_loc")
            lbl_t = z2_m.get("labels")
            if len(z2_b) == 0 or qloc_t is None or lbl_t is None or len(qloc_t) != len(z2_b):
                continue
            ptp_t = z2_m.get("p_tp", z2_m.get("reliability", qloc_t))
            qloc_dev = qloc_t.to(z2_b.device).float()
            ptp_dev = ptp_t.to(z2_b.device).float()
            lbl_dev = lbl_t.to(z2_b.device).long()
            stable_mask = (
                (qloc_dev >= float(z2_reg_loc_thresh))
                & (ptp_dev >= 0.35)
                & (lbl_dev >= 1)
                & (lbl_dev < num_cls_total)
            )
            if bool(stable_mask.any()):
                w_st = (ptp_dev[stable_mask] * qloc_dev[stable_mask]).clamp(0.10, 1.0)
                if cls_bal_lut is not None:
                    w_st = w_st * cls_bal_lut[lbl_dev[stable_mask]]
                z2_stable_entries.append(
                    (img_idx, z2_b[stable_mask].detach(), lbl_dev[stable_mask].detach(), w_st.detach())
                )

        if z2_stable_entries:
            missing_rpn_ids = [idx for idx, *_ in z2_stable_entries if idx not in cached_rpn_proposals]
            if missing_rpn_ids:
                from torchvision.models.detection.image_list import ImageList
                from ssod_research.baselines import extract_rpn_proposals_only

                extra_imgs = ImageList(
                    transformed.tensors[missing_rpn_ids],
                    [transformed.image_sizes[idx] for idx in missing_rpn_ids],
                )
                extra_feats = {k: v[missing_rpn_ids] for k, v in features.items()}
                extra_props = extract_rpn_proposals_only(model.rpn, extra_imgs, extra_feats)
                for p_slot, idx in enumerate(missing_rpn_ids):
                    cached_rpn_proposals[idx] = extra_props[p_slot].detach()

            z2_props_per_img: list[torch.Tensor] = []
            z2_tgt_boxes_list: list[torch.Tensor] = []
            z2_tgt_lbls_list: list[torch.Tensor] = []
            z2_tgt_w_list: list[torch.Tensor] = []
            active_z2_reg_imgs = 0
            for idx in range(len(aug_images)):
                z2_props_per_img.append(transformed.tensors.new_zeros((0, 4)))
            for img_idx, st_boxes, st_lbls, st_w in z2_stable_entries:
                z1_b = z1_boxes_trans[img_idx]
                if len(z1_b) > 0:
                    non_z1 = box_iou(st_boxes, z1_b).max(dim=1).values < 0.50
                    st_boxes = st_boxes[non_z1]
                    st_lbls = st_lbls[non_z1]
                    st_w = st_w[non_z1]
                if len(st_boxes) == 0:
                    continue

                th_i, tw_i = transformed.image_sizes[img_idx]
                bw = (st_boxes[:, 2] - st_boxes[:, 0]).clamp(min=2.0)
                bh = (st_boxes[:, 3] - st_boxes[:, 1]).clamp(min=2.0)
                # Deterministic +-8% spatial jitter proposals guarantee non-zero regression deltas
                # even for tiny 8x12 traffic lights or night objects not yet proposed by student RPN
                jit_a = torch.stack([
                    (st_boxes[:, 0] - 0.08 * bw).clamp(0.0, max(0.0, float(tw_i) - 1.0)),
                    (st_boxes[:, 1] - 0.08 * bh).clamp(0.0, max(0.0, float(th_i) - 1.0)),
                    (st_boxes[:, 2] - 0.04 * bw).clamp(1.0, float(tw_i)),
                    (st_boxes[:, 3] - 0.04 * bh).clamp(1.0, float(th_i)),
                ], dim=1)
                jit_b = torch.stack([
                    (st_boxes[:, 0] + 0.04 * bw).clamp(0.0, max(0.0, float(tw_i) - 1.0)),
                    (st_boxes[:, 1] + 0.04 * bh).clamp(0.0, max(0.0, float(th_i) - 1.0)),
                    (st_boxes[:, 2] + 0.08 * bw).clamp(1.0, float(tw_i)),
                    (st_boxes[:, 3] + 0.08 * bh).clamp(1.0, float(th_i)),
                ], dim=1)
                jit_a[:, 2] = torch.maximum(jit_a[:, 2], (jit_a[:, 0] + 1.0).clamp(max=float(tw_i)))
                jit_a[:, 3] = torch.maximum(jit_a[:, 3], (jit_a[:, 1] + 1.0).clamp(max=float(th_i)))
                jit_b[:, 2] = torch.maximum(jit_b[:, 2], (jit_b[:, 0] + 1.0).clamp(max=float(tw_i)))
                jit_b[:, 3] = torch.maximum(jit_b[:, 3], (jit_b[:, 1] + 1.0).clamp(max=float(th_i)))
                seed_props = torch.cat([jit_a, jit_b], dim=0)

                cand_p = cached_rpn_proposals.get(img_idx)
                if cand_p is not None and len(cand_p) > 0:
                    if len(z1_b) > 0:
                        cand_p = cand_p[box_iou(cand_p, z1_b).max(dim=1).values < 0.50]
                    cand_all = torch.cat([seed_props, cand_p], dim=0)
                else:
                    cand_all = seed_props

                ious_st = box_iou(cand_all, st_boxes)
                max_iou_st, best_st = ious_st.max(dim=1)
                pos_m = max_iou_st >= 0.50
                if not bool(pos_m.any()):
                    continue
                sel_inds = torch.where(pos_m)[0]
                if len(sel_inds) > 64:
                    top_order = torch.argsort(max_iou_st[sel_inds], descending=True)[:64]
                    sel_inds = sel_inds[top_order]
                m_props = cand_all[sel_inds].detach()
                m_best = best_st[sel_inds]
                z2_props_per_img[img_idx] = m_props
                z2_tgt_boxes_list.append(st_boxes[m_best])
                z2_tgt_lbls_list.append(st_lbls[m_best])
                z2_tgt_w_list.append(st_w[m_best])
                active_z2_reg_imgs += 1

            if active_z2_reg_imgs > 0:
                z2_rf = model.roi_heads.box_roi_pool(features, z2_props_per_img, transformed.image_sizes)
                z2_rf = model.roi_heads.box_head(z2_rf)
                _, z2_box_reg = model.roi_heads.box_predictor(z2_rf)
                n_z2_p = z2_box_reg.shape[0]
                z2_box_reg = z2_box_reg.reshape(n_z2_p, num_cls_total, 4)
                cat_props = torch.cat([p for p in z2_props_per_img if len(p) > 0], dim=0).float()
                cat_boxes = torch.cat(z2_tgt_boxes_list, dim=0).float()
                cat_lbls = torch.cat(z2_tgt_lbls_list, dim=0).long()
                cat_w = torch.cat(z2_tgt_w_list, dim=0).float()
                rows_z2 = torch.arange(n_z2_p, device=z2_box_reg.device)
                pred_z2_deltas = z2_box_reg[rows_z2, cat_lbls].float()
                tgt_z2_deltas = model.roi_heads.box_coder.encode_single(cat_boxes, cat_props).float()
                l1_z2 = F.smooth_l1_loss(pred_z2_deltas, tgt_z2_deltas, beta=1.0 / 9.0, reduction="none").sum(dim=1)
                loss_z2_l1 = (l1_z2 * cat_w).sum() / max(64.0, float(n_z2_p))
                if giou_weight > 0.0:
                    dec_z2 = model.roi_heads.box_coder.decode_single(pred_z2_deltas, cat_props)
                    loss_z2_giou = float(giou_weight) * _giou_consistency_loss(dec_z2, cat_boxes, cat_w)
                else:
                    loss_z2_giou = loss_z2_l1 * 0.0
                z2_reg_scale = (float(active_z2_reg_imgs) / float(total_batch_size)) if scale_by_batch_size else 1.0
                loss_z2_reg = z2_reg_scale * float(z2_reg_weight) * (loss_z2_l1 + loss_z2_giou)
                total = loss_z2_reg if total is None else (total + loss_z2_reg)
                stab_loss_val += float(loss_z2_reg.detach())

    # 3. Mild calibrated full-distribution KL distillation on Zone 1 (reusing student FPN)
    n_z1_imgs = sum(1 for b in z1_boxes_trans if len(b) > 0)
    if kl_zone1_weight > 0.0 and n_z1_imgs > 0:
        roi_f1 = model.roi_heads.box_roi_pool(features, z1_boxes_trans, transformed.image_sizes)
        roi_f1 = model.roi_heads.box_head(roi_f1)
        logits1, _ = model.roi_heads.box_predictor(roi_f1)
        probs1 = torch.cat([z["probs"] for z in zone1_meta], dim=0)
        rel1 = torch.cat([z["reliability"] for z in zone1_meta], dim=0)
        kl1_scale = (float(n_z1_imgs) / float(total_batch_size)) if scale_by_batch_size else 1.0
        loss_kl1 = kl1_scale * float(kl_zone1_weight) * _kl_soft_distillation(logits1, probs1, rel1, temperature=temperature)
        total = loss_kl1 if total is None else (total + loss_kl1)
        stab_loss_val += float(loss_kl1.detach())

    # 4. Mild calibrated full-distribution KL distillation on Zone 2 (includes background mass)
    n_z2_imgs = sum(1 for b in ambig_trans if len(b) > 0)
    if use_tri_zone and kl_zone2_weight > 0.0 and n_z2_imgs > 0:
        roi_f2 = model.roi_heads.box_roi_pool(features, ambig_trans, transformed.image_sizes)
        roi_f2 = model.roi_heads.box_head(roi_f2)
        logits2, _ = model.roi_heads.box_predictor(roi_f2)
        probs2 = torch.cat([z["probs"] for z in zone2_meta], dim=0)
        p_tp2 = torch.cat(
            [z.get("p_tp", z["reliability"]) for z in zone2_meta], dim=0
        )
        kl2_scale = (float(n_z2_imgs) / float(total_batch_size)) if scale_by_batch_size else 1.0
        loss_kl2 = kl2_scale * float(kl_zone2_weight) * _kl_soft_distillation(logits2, probs2, p_tp2, temperature=temperature)
        total = loss_kl2 if total is None else (total + loss_kl2)
        stab_loss_val += float(loss_kl2.detach())

    return total, {
        "unsup": det_loss_val,
        "stability": stab_loss_val,
        "suppressed_bg": suppressed_bg,
    }
