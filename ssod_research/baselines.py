"""Faithful native TorchVision implementations of strong published SSOD baselines.

Provides apples-to-apples comparisons under the exact same backbone, image
resolution, optimizer, LR schedule, and supervised warm-start checkpoint:

1. ``soft_teacher`` (Xu et al., ICCV 2021):
   - Box Jittering localization reliability estimation on teacher FPN features.
   - Soft-Negative Background Classification Weighting: background ROI proposals
     are weighted by normalized EMA Teacher background probability r_j = p_T(bg)
     so missed foreground objects are not crushed as hard negatives.
2. ``consistent_teacher`` (Wang et al., CVPR 2023 — dynamic GMM thresholding):
   - Online 2-component 1D Gaussian Mixture Model (EM) per class separating
     high-confidence foreground from low-confidence noise without fixed cutoffs.
   - Checkpoint-serializable sliding score buffers for exact deterministic resume.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ssod_research.care_ssod import _flip_boxes_x


# ------------------------------------------------------------------ 1. Soft Teacher (ICCV 2021)
@torch.no_grad()
def soft_teacher_extract(
    teacher,
    images: list[torch.Tensor],
    *,
    cls_threshold: float = 0.70,
    reg_threshold: float = 0.50,
    jitter_times: int = 4,
    jitter_scale: float = 0.06,
    jitter_thresh: float = 0.04,
    rng=None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run EMA Teacher once, cache FPN features, and score boxes via Box Jittering."""
    teacher.eval()
    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
    transformed, _ = teacher.transform(images, None)
    t_features = teacher.backbone(transformed.tensors)
    if isinstance(t_features, torch.Tensor):
        t_features = {"0": t_features}
    proposals, _ = teacher.rpn(transformed, t_features, None)
    detections, _ = teacher.roi_heads(t_features, proposals, transformed.image_sizes, None)
    detections = teacher.transform.postprocess(detections, transformed.image_sizes, original_sizes)

    results: list[dict[str, Any]] = []
    stats = {"candidates": 0, "accepted": 0, "reg_accepted": 0, "quality_sum": 0.0}

    for img_idx, (det, orig_sz, trans_sz) in enumerate(zip(detections, original_sizes, transformed.image_sizes)):
        boxes = det["boxes"]
        labels = det["labels"]
        scores = det["scores"]
        stats["candidates"] += len(scores)

        keep_mask = torch.isfinite(scores) & (scores >= min(cls_threshold, reg_threshold))
        cand_boxes = boxes[keep_mask]
        cand_labels = labels[keep_mask]
        cand_scores = scores[keep_mask]

        if len(cand_boxes) == 0:
            results.append({
                "boxes": boxes.new_zeros((0, 4)),
                "labels": labels.new_zeros((0,), dtype=torch.long),
                "scores": scores.new_zeros((0,)),
                "reg_valid": torch.zeros((0,), dtype=torch.bool, device=boxes.device),
                "jitter_std": scores.new_zeros((0,)),
            })
            continue

        # Scale candidate boxes to teacher's transformed image coordinates for box jittering
        oh, ow = orig_sz
        th, tw = trans_sz
        sy = float(th) / max(1, oh)
        sx = float(tw) / max(1, ow)
        scaled = cand_boxes.clone()
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy

        w = (scaled[:, 2] - scaled[:, 0]).clamp(min=1.0)
        h = (scaled[:, 3] - scaled[:, 1]).clamp(min=1.0)

        # Jitter K times and measure normalized coordinate std after Teacher ROI regression
        refined_list = []
        single_feat = {k: v[img_idx:img_idx + 1] for k, v in t_features.items()}
        for _ in range(max(2, int(jitter_times))):
            if rng is not None:
                noise_vals = [rng.uniform(-jitter_scale, jitter_scale) for _ in range(len(scaled) * 4)]
                noise = torch.tensor(noise_vals, device=scaled.device, dtype=scaled.dtype).reshape(-1, 4)
            else:
                noise = (torch.rand((len(scaled), 4), device=scaled.device, dtype=scaled.dtype) * 2.0 - 1.0) * jitter_scale
            jittered = scaled.clone()
            jittered[:, 0] = (scaled[:, 0] + noise[:, 0] * w).clamp(0.0, float(tw))
            jittered[:, 1] = (scaled[:, 1] + noise[:, 1] * h).clamp(0.0, float(th))
            jittered[:, 2] = (scaled[:, 2] + noise[:, 2] * w).clamp(0.0, float(tw))
            jittered[:, 3] = (scaled[:, 3] + noise[:, 3] * h).clamp(0.0, float(th))
            # Ensure x2 > x1 and y2 > y1
            jittered[:, 2] = torch.maximum(jittered[:, 2], jittered[:, 0] + 1.0)
            jittered[:, 3] = torch.maximum(jittered[:, 3], jittered[:, 1] + 1.0)

            roi_f = teacher.roi_heads.box_roi_pool(single_feat, [jittered], [trans_sz])
            roi_f = teacher.roi_heads.box_head(roi_f)
            _, reg_deltas = teacher.roi_heads.box_predictor(roi_f)
            decoded = teacher.roi_heads.box_coder.decode(reg_deltas, [jittered])
            rows = torch.arange(len(cand_labels), device=cand_labels.device)
            refined = decoded[rows, cand_labels.long()]
            refined_list.append(refined)

        stacked = torch.stack(refined_list, dim=0)  # (K, N, 4)
        coord_std = stacked.std(dim=0, unbiased=False)
        norm_std = 0.25 * (
            (coord_std[:, 0] + coord_std[:, 2]) / w
            + (coord_std[:, 1] + coord_std[:, 3]) / h
        )

        cls_keep = cand_scores >= cls_threshold
        reg_keep = (cand_scores >= reg_threshold) & (norm_std <= jitter_thresh)
        final_keep = cls_keep | reg_keep

        out_boxes = cand_boxes[final_keep].detach()
        out_labels = cand_labels[final_keep].detach()
        out_scores = cand_scores[final_keep].detach()
        out_reg = reg_keep[final_keep].detach()
        out_std = norm_std[final_keep].detach()

        stats["accepted"] += len(out_boxes)
        stats["reg_accepted"] += int(out_reg.sum().item())
        stats["quality_sum"] += float(out_scores.sum().item())
        results.append({
            "boxes": out_boxes,
            "labels": out_labels,
            "scores": out_scores,
            "reg_valid": out_reg,
            "jitter_std": out_std,
        })

    teacher_cache = {
        "features": t_features,
        "transformed_sizes": transformed.image_sizes,
        "original_sizes": original_sizes,
    }
    return results, stats, teacher_cache


def soft_teacher_unsupervised_loss(
    model,
    teacher,
    aug_images: list[torch.Tensor],
    aug_targets: list[dict[str, torch.Tensor]],
    flips: list[bool],
    reg_valid_list: list[torch.Tensor],
    teacher_cache: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Soft Teacher (ICCV 2021) unsupervised loss with soft-negative bg weighting."""
    kept = [i for i, target in enumerate(aug_targets) if len(target["boxes"])]
    if not kept:
        return None, {"unsup": 0.0, "stability": 0.0}

    if len(kept) != len(aug_images):
        aug_images = [aug_images[i] for i in kept]
        aug_targets = [aug_targets[i] for i in kept]
        flips = [flips[i] for i in kept]
        reg_valid_list = [reg_valid_list[i] for i in kept]
        teacher_cache = {
            "features": {k: v[kept] for k, v in teacher_cache["features"].items()},
            "transformed_sizes": [teacher_cache["transformed_sizes"][i] for i in kept],
        }

    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in aug_images]
    transformed, transformed_targets = model.transform(aug_images, aug_targets)
    s_features = model.backbone(transformed.tensors)
    if isinstance(s_features, torch.Tensor):
        s_features = {"0": s_features}

    proposals, rpn_losses = model.rpn(transformed, s_features, transformed_targets)
    sampled_props, matched_idxs, labels_list, reg_targets_list = model.roi_heads.select_training_samples(
        proposals, transformed_targets
    )

    # Compute student ROI logits & regressions
    box_features = model.roi_heads.box_roi_pool(s_features, sampled_props, transformed.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    # Map student sampled proposals back to weak teacher coordinates to obtain soft bg weights r_j
    t_props_list = []
    for props, was_flipped, s_sz, o_sz, t_sz in zip(
        sampled_props,
        flips,
        transformed.image_sizes,
        original_sizes,
        teacher_cache["transformed_sizes"],
    ):
        p = props.detach().clone()
        # Student transformed -> original coords
        p[:, [0, 2]] *= float(o_sz[1]) / max(1, s_sz[1])
        p[:, [1, 3]] *= float(o_sz[0]) / max(1, s_sz[0])
        if was_flipped:
            p = _flip_boxes_x(p, o_sz[1])
        # Original -> teacher transformed coords
        p[:, [0, 2]] *= float(t_sz[1]) / max(1, o_sz[1])
        p[:, [1, 3]] *= float(t_sz[0]) / max(1, o_sz[0])
        t_props_list.append(p)

    with torch.no_grad():
        t_roi_f = teacher.roi_heads.box_roi_pool(
            teacher_cache["features"], t_props_list, teacher_cache["transformed_sizes"]
        )
        t_roi_f = teacher.roi_heads.box_head(t_roi_f)
        t_logits, _ = teacher.roi_heads.box_predictor(t_roi_f)
        t_bg_probs = F.softmax(t_logits.float(), dim=-1)[:, 0].detach()  # column 0 = background

    labels = torch.cat(labels_list, dim=0)
    regression_targets = torch.cat(reg_targets_list, dim=0)

    # Eq. 5 of Soft Teacher: foreground weight = 1, background weight = r_j / mean(r_bg)
    per_sample_cls = F.cross_entropy(class_logits.float(), labels, reduction="none")
    fg_mask = labels > 0
    bg_mask = labels == 0
    weights = torch.ones_like(per_sample_cls)
    if bool(bg_mask.any()):
        bg_r = t_bg_probs[bg_mask].clamp(min=1e-4)
        weights[bg_mask] = bg_r / bg_r.mean().clamp(min=1e-4)
    loss_cls = (per_sample_cls * weights).sum() / weights.sum().clamp(min=1.0)

    # Box regression only on foreground proposals matched to jitter-verified boxes
    reg_keep_per_prop = []
    for m_idx, r_valid in zip(matched_idxs, reg_valid_list):
        if len(r_valid) == 0:
            reg_keep_per_prop.append(torch.zeros(len(m_idx), dtype=torch.bool, device=m_idx.device))
        else:
            clamped = m_idx.clamp(min=0, max=len(r_valid) - 1)
            reg_keep_per_prop.append(r_valid[clamped])
    reg_valid_mask = fg_mask & torch.cat(reg_keep_per_prop, dim=0)

    sampled_pos_inds = torch.where(reg_valid_mask)[0]
    if len(sampled_pos_inds) > 0:
        n_total, num_classes = class_logits.shape
        box_reg = box_regression.reshape(n_total, num_classes, 4)
        labels_pos = labels[sampled_pos_inds]
        loss_box = F.smooth_l1_loss(
            box_reg[sampled_pos_inds, labels_pos],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9.0,
            reduction="sum",
        ) / max(1, labels.numel())
    else:
        loss_box = box_regression.sum() * 0.0

    rpn_total = sum(rpn_losses.values())
    total = rpn_total + loss_cls + loss_box
    return total, {"unsup": float(total.detach()), "stability": 0.0}


# ------------------------------------------------------------------ 2. Dynamic GMM (Consistent-Teacher style)
def _fit_1d_two_component_gmm(
    values: list[float],
    *,
    max_iter: int = 15,
    eps: float = 1e-6,
) -> tuple[float, float, float, float]:
    """Fit a deterministic 2-component 1D GMM via EM; return (mu_lo, sd_lo, mu_hi, sd_hi)."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    q25 = xs[n // 4]
    q75 = xs[(3 * n) // 4]
    mu0, mu1 = q25, max(q75, q25 + 0.05)
    var0 = var1 = 0.04
    pi0 = pi1 = 0.5

    def _gauss(x: float, mu: float, var: float) -> float:
        s = math.sqrt(max(var, 1e-4))
        z = (x - mu) / s
        return math.exp(-0.5 * min(50.0, z * z)) / (s * 2.5066282746310002)

    for _ in range(max_iter):
        g0_sum = g1_sum = 0.0
        x0_sum = x1_sum = 0.0
        gammas = []
        for x in xs:
            p0 = pi0 * _gauss(x, mu0, var0) + eps
            p1 = pi1 * _gauss(x, mu1, var1) + eps
            denom = p0 + p1
            r1 = p1 / denom
            r0 = 1.0 - r1
            gammas.append((r0, r1))
            g0_sum += r0
            g1_sum += r1
            x0_sum += r0 * x
            x1_sum += r1 * x
        if g0_sum < 1e-3 or g1_sum < 1e-3:
            break
        mu0 = x0_sum / g0_sum
        mu1 = x1_sum / g1_sum
        var0 = sum(r0 * (x - mu0) ** 2 for x, (r0, _) in zip(xs, gammas)) / g0_sum + 1e-4
        var1 = sum(r1 * (x - mu1) ** 2 for x, (_, r1) in zip(xs, gammas)) / g1_sum + 1e-4
        pi0 = g0_sum / n
        pi1 = g1_sum / n

    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
        var0, var1 = var1, var0
    return mu0, math.sqrt(var0), mu1, math.sqrt(var1)


class DynamicGMMThreshold:
    """Consistent-Teacher-style per-class 1D GMM adaptive threshold (checkpoint-safe)."""

    def __init__(
        self,
        num_foreground_classes: int = 10,
        *,
        base_threshold: float = 0.70,
        buffer_size: int = 256,
        min_samples: int = 20,
        min_threshold: float = 0.35,
        max_threshold: float = 0.90,
    ):
        self.num_classes = int(num_foreground_classes)
        self.base_threshold = float(base_threshold)
        self.buffer_size = int(buffer_size)
        self.min_samples = int(min_samples)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.buffers: dict[int, list[float]] = {c: [] for c in range(1, self.num_classes + 1)}
        self.thresholds: dict[int, float] = {c: float(base_threshold) for c in range(1, self.num_classes + 1)}

    def update_and_fit(self, labels: list[int], scores: list[float]) -> None:
        touched: set[int] = set()
        for cls_id, sc in zip(labels, scores):
            cls_id = int(cls_id)
            if cls_id in self.buffers and 0.05 <= float(sc) <= 1.0:
                buf = self.buffers[cls_id]
                buf.append(float(sc))
                if len(buf) > self.buffer_size:
                    del buf[: len(buf) - self.buffer_size]
                touched.add(cls_id)
        for cls_id in touched:
            buf = self.buffers[cls_id]
            if len(buf) >= self.min_samples:
                _, _, mu_hi, sd_hi = _fit_1d_two_component_gmm(buf)
                tau = max(self.min_threshold, min(self.max_threshold, mu_hi - 0.5 * sd_hi))
                self.thresholds[cls_id] = round(float(tau), 4)

    def threshold_for(self, cls_id: int) -> float:
        return float(self.thresholds.get(int(cls_id), self.base_threshold))

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "base_threshold": self.base_threshold,
            "buffer_size": self.buffer_size,
            "min_samples": self.min_samples,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "buffers": {int(c): [float(x) for x in vals] for c, vals in self.buffers.items()},
            "thresholds": {int(c): float(v) for c, v in self.thresholds.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "buffers" in state:
            self.buffers = {int(c): [float(x) for x in vals] for c, vals in state["buffers"].items()}
        if "thresholds" in state:
            self.thresholds = {int(c): float(v) for c, v in state["thresholds"].items()}
