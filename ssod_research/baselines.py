"""Faithful native TorchVision implementations of strong published SSOD baselines.

Provides apples-to-apples comparisons under the exact same backbone, image
resolution, optimizer, LR schedule, and supervised warm-start checkpoint:

1. ``soft_teacher`` (Xu et al., ICCV 2021):
   - Strictly decoupled classification vs regression pseudo-label filtering:
     * Classification & RPN targets use high-confidence ``cls_valid`` boxes
       (``score >= cls_threshold``), preventing low-score false positives from
       poisoning RPN/ROI classification.
     * Box regression targets use Box-Jittering variance-verified ``reg_valid``
       boxes (``score >= reg_threshold`` and normalized regression std
       ``<= jitter_thresh``).
   - Soft-Negative Background Classification Weighting (Eq. 5 of Soft Teacher):
     background ROI proposals are weighted by normalized EMA Teacher background
     probability ``r_j = p_T(bg) / mean(p_T(bg))`` so missed foreground objects
     are not crushed as hard negatives.
2. ``consistent_teacher`` (Wang et al., CVPR 2023 — dynamic GMM thresholding):
   - Online 2-component 1D Gaussian Mixture Model (EM) per class on non-trivial
     candidate scores (``>= 0.15``) with Bayes-optimal Gaussian posterior
     decision boundary separating foreground from background noise.
   - Checkpoint-serializable sliding score buffers for exact deterministic resume.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.models.detection.rpn import concat_box_prediction_layers

from ssod_research.care_ssod import _flip_boxes_x


@torch.no_grad()
def extract_rpn_proposals_only(rpn, images, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Extract RPN proposals without targets while preserving training pre/post-NMS limits (2000).

    Calling ``rpn(images, features, None)`` with ``rpn.training=True`` raises
    ``ValueError('targets should not be None')`` in TorchVision, whereas toggling
    ``rpn.training=False`` halves ``pre_nms_top_n`` and ``post_nms_top_n`` (2000 -> 1000).
    """
    for hook in rpn._forward_pre_hooks.values():
        res = hook(rpn, (images, features, None))
        if res is not None:
            images, features, _ = res
    feat_list = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(feat_list)
    anchors = rpn.anchor_generator(images, feat_list)
    num_images = len(anchors)
    num_anchors_per_level = [o[0].shape[0] * o[0].shape[1] * o[0].shape[2] for o in objectness]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, _ = rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
    return [b.detach() for b in boxes]


# ------------------------------------------------------------------ 1. Soft Teacher (ICCV 2021)
@torch.no_grad()
def soft_teacher_extract(
    teacher,
    images: list[torch.Tensor],
    *,
    cls_threshold: float = 0.70,
    reg_threshold: float = 0.50,
    jitter_times: int = 6,
    jitter_scale: float = 0.06,
    jitter_thresh: float = 0.03,
    rng=None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run EMA Teacher once, cache FPN features, and score boxes via Box Jittering.

    Strictly separates ``cls_valid`` (score >= cls_threshold) from ``reg_valid``
    (score >= reg_threshold & norm_std <= jitter_thresh) as in Xu et al., ICCV 2021.
    """
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
                "cls_valid": torch.zeros((0,), dtype=torch.bool, device=boxes.device),
                "reg_valid": torch.zeros((0,), dtype=torch.bool, device=boxes.device),
                "jitter_std": scores.new_zeros((0,)),
            })
            continue

        oh, ow = orig_sz
        th, tw = trans_sz
        sy = float(th) / max(1, oh)
        sx = float(tw) / max(1, ow)
        scaled = cand_boxes.clone()
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy

        w = (scaled[:, 2] - scaled[:, 0]).clamp(min=1.0)
        h = (scaled[:, 3] - scaled[:, 1]).clamp(min=1.0)

        refined_list = []
        single_feat = {k: v[img_idx:img_idx + 1] for k, v in t_features.items()}
        k_jit = max(2, int(jitter_times))
        for _ in range(k_jit):
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
            jittered[:, 2] = torch.maximum(jittered[:, 2], jittered[:, 0] + 1.0)
            jittered[:, 3] = torch.maximum(jittered[:, 3], jittered[:, 1] + 1.0)

            roi_f = teacher.roi_heads.box_roi_pool(single_feat, [jittered], [trans_sz])
            roi_f = teacher.roi_heads.box_head(roi_f)
            _, reg_deltas = teacher.roi_heads.box_predictor(roi_f)
            decoded = teacher.roi_heads.box_coder.decode(reg_deltas.float(), [jittered.float()])
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
        out_cls = cls_keep[final_keep].detach()
        out_reg = reg_keep[final_keep].detach()
        out_std = norm_std[final_keep].detach()

        stats["accepted"] += int(out_cls.sum().item())
        stats["reg_accepted"] += int(out_reg.sum().item())
        stats["quality_sum"] += float(out_scores[out_cls].sum().item()) if bool(out_cls.any()) else 0.0
        results.append({
            "boxes": out_boxes,
            "labels": out_labels,
            "scores": out_scores,
            "cls_valid": out_cls,
            "reg_valid": out_reg,
            "jitter_std": out_std,
        })

    teacher_cache = {
        "features": {k: v.detach() for k, v in t_features.items()},
        "transformed_sizes": list(transformed.image_sizes),
        "original_sizes": original_sizes,
    }
    return results, stats, teacher_cache


def _slice_target_by_mask(target: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {
        "boxes": target["boxes"][mask],
        "labels": target["labels"][mask],
    }
    if "image_id" in target:
        out["image_id"] = target["image_id"]
    b = out["boxes"]
    out["area"] = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=0.0)
    out["iscrowd"] = torch.zeros(len(b), dtype=torch.long, device=b.device)
    return out


def soft_teacher_unsupervised_loss(
    model,
    teacher,
    aug_images: list[torch.Tensor],
    aug_targets: list[dict[str, torch.Tensor]],
    flips: list[bool],
    reg_valid_list: list[torch.Tensor],
    teacher_cache: dict[str, Any],
    *,
    cls_valid_list: list[torch.Tensor] | None = None,
    scale_by_batch_size: bool = True,
    filter_zero_box_images: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Faithful Soft Teacher (ICCV 2021) unsupervised loss with decoupled cls/reg & soft-negatives."""
    total_batch_size = max(1, len(aug_images))
    if cls_valid_list is None:
        cls_targets = aug_targets
    else:
        cls_targets = [
            _slice_target_by_mask(t, m.to(t["boxes"].device))
            for t, m in zip(aug_targets, cls_valid_list)
        ]
    reg_targets = [
        _slice_target_by_mask(t, m.to(t["boxes"].device))
        for t, m in zip(aug_targets, reg_valid_list)
    ]

    if filter_zero_box_images:
        active = [
            i for i in range(len(aug_images))
            if len(cls_targets[i]["boxes"]) > 0 or len(reg_targets[i]["boxes"]) > 0
        ]
    else:
        active = list(range(len(aug_images)))
    if not active:
        return None, {"unsup": 0.0, "stability": 0.0, "active_cls_images": 0, "active_reg_images": 0}

    if len(active) != len(aug_images):
        aug_images = [aug_images[i] for i in active]
        cls_targets = [cls_targets[i] for i in active]
        reg_targets = [reg_targets[i] for i in active]
        flips = [flips[i] for i in active]
        teacher_cache = {
            "features": {k: v[active] for k, v in teacher_cache["features"].items()},
            "transformed_sizes": [teacher_cache["transformed_sizes"][i] for i in active],
        }

    cls_pos = (
        [i for i, t in enumerate(cls_targets) if len(t["boxes"]) > 0]
        if filter_zero_box_images
        else list(range(len(cls_targets)))
    )
    reg_pos = [i for i, t in enumerate(reg_targets) if len(t["boxes"]) > 0]
    has_cls = len(cls_pos) > 0 and (not filter_zero_box_images or any(len(t["boxes"]) > 0 for t in cls_targets))
    has_reg = len(reg_pos) > 0

    original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in aug_images]
    transformed, trans_cls_targets = model.transform(aug_images, cls_targets if has_cls else None)
    s_features = model.backbone(transformed.tensors)
    if isinstance(s_features, torch.Tensor):
        s_features = {"0": s_features}

    total = None
    proposals_by_idx: dict[int, torch.Tensor] = {}

    # 1. RPN + Soft-Negative Classification ONLY on images with >= 1 cls_valid box
    if has_cls and trans_cls_targets is not None:
        if len(cls_pos) == len(aug_images):
            cls_images = transformed
            cls_s_features = s_features
            cls_t_sub = trans_cls_targets
            cls_flips = flips
            cls_orig_sizes = original_sizes
            cls_t_cache = teacher_cache
        else:
            from torchvision.models.detection.image_list import ImageList

            cls_images = ImageList(
                transformed.tensors[cls_pos],
                [transformed.image_sizes[i] for i in cls_pos],
            )
            cls_s_features = {k: v[cls_pos] for k, v in s_features.items()}
            cls_t_sub = [trans_cls_targets[i] for i in cls_pos]
            cls_flips = [flips[i] for i in cls_pos]
            cls_orig_sizes = [original_sizes[i] for i in cls_pos]
            cls_t_cache = {
                "features": {k: v[cls_pos] for k, v in teacher_cache["features"].items()},
                "transformed_sizes": [teacher_cache["transformed_sizes"][i] for i in cls_pos],
            }

        proposals_cls, rpn_losses = model.rpn(cls_images, cls_s_features, cls_t_sub)
        for idx, prop in zip(cls_pos, proposals_cls):
            proposals_by_idx[idx] = prop

        sampled_props, _, labels_list, _ = model.roi_heads.select_training_samples(
            proposals_cls, cls_t_sub
        )
        box_features = model.roi_heads.box_roi_pool(cls_s_features, sampled_props, cls_images.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, _ = model.roi_heads.box_predictor(box_features)

        # Map student sampled proposals back to weak teacher coordinates for soft bg weights r_j
        t_props_list = []
        for props, was_flipped, s_sz, o_sz, t_sz in zip(
            sampled_props,
            cls_flips,
            cls_images.image_sizes,
            cls_orig_sizes,
            cls_t_cache["transformed_sizes"],
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
                cls_t_cache["features"], t_props_list, cls_t_cache["transformed_sizes"]
            )
            t_roi_f = teacher.roi_heads.box_head(t_roi_f)
            t_logits, _ = teacher.roi_heads.box_predictor(t_roi_f)
            t_bg_probs = F.softmax(t_logits.float(), dim=-1)[:, 0].detach()

        labels = torch.cat(labels_list, dim=0)
        per_sample_cls = F.cross_entropy(class_logits.float(), labels, reduction="none")
        bg_mask = labels == 0
        weights = torch.ones_like(per_sample_cls)
        if bool(bg_mask.any()):
            bg_r = t_bg_probs[bg_mask].clamp(min=1e-4)
            weights[bg_mask] = bg_r / bg_r.mean().clamp(min=1e-4)
        loss_cls = (per_sample_cls * weights).sum() / weights.sum().clamp(min=1.0)
        rpn_total = sum(rpn_losses.values())
        cls_scale = (float(len(cls_pos)) / float(total_batch_size)) if scale_by_batch_size else 1.0
        total = cls_scale * (rpn_total + loss_cls)

    # 2. Box regression ONLY on foreground proposals matched to jitter-verified reg_valid targets
    if has_reg:
        missing_reg = [i for i in reg_pos if i not in proposals_by_idx]
        if missing_reg:
            from torchvision.models.detection.image_list import ImageList

            extra_imgs = ImageList(
                transformed.tensors[missing_reg],
                [transformed.image_sizes[i] for i in missing_reg],
            )
            extra_feats = {k: v[missing_reg] for k, v in s_features.items()}
            extra_props = extract_rpn_proposals_only(model.rpn, extra_imgs, extra_feats)
            for idx, prop in zip(missing_reg, extra_props):
                proposals_by_idx[idx] = prop

        reg_props_in = [proposals_by_idx[i] for i in reg_pos]
        trans_reg_targets = []
        for i in reg_pos:
            tgt = reg_targets[i]
            oh, ow = original_sizes[i]
            th, tw = transformed.image_sizes[i]
            b = tgt["boxes"].clone()
            b[:, [0, 2]] *= float(tw) / max(1, ow)
            b[:, [1, 3]] *= float(th) / max(1, oh)
            trans_reg_targets.append({"boxes": b, "labels": tgt["labels"]})

        reg_s_features = (
            s_features if len(reg_pos) == len(aug_images) else {k: v[reg_pos] for k, v in s_features.items()}
        )
        reg_img_sizes = [transformed.image_sizes[i] for i in reg_pos]
        reg_props, _, reg_labels_list, reg_gt_deltas_list = model.roi_heads.select_training_samples(
            reg_props_in, trans_reg_targets
        )
        reg_labels = torch.cat(reg_labels_list, dim=0)
        reg_gt_deltas = torch.cat(reg_gt_deltas_list, dim=0)
        pos_inds = torch.where(reg_labels > 0)[0]
        if len(pos_inds) > 0:
            pos_props_per_img = [p_img[l_img > 0] for p_img, l_img in zip(reg_props, reg_labels_list)]
            pos_feats = model.roi_heads.box_roi_pool(reg_s_features, pos_props_per_img, reg_img_sizes)
            pos_feats = model.roi_heads.box_head(pos_feats)
            pos_logits, pos_box_reg = model.roi_heads.box_predictor(pos_feats)
            n_pos, num_classes = pos_logits.shape
            pos_box_reg = pos_box_reg.reshape(n_pos, num_classes, 4)
            labels_pos = reg_labels[pos_inds]
            reg_scale = (float(len(reg_pos)) / float(total_batch_size)) if scale_by_batch_size else 1.0
            loss_box = reg_scale * (
                F.smooth_l1_loss(
                    pos_box_reg[torch.arange(n_pos, device=labels_pos.device), labels_pos].float(),
                    reg_gt_deltas[pos_inds].float(),
                    beta=1.0 / 9.0,
                    reduction="sum",
                )
                / max(1, reg_labels.numel())
            )
            total = loss_box if total is None else (total + loss_box)

    if total is None:
        return None, {"unsup": 0.0, "stability": 0.0, "active_cls_images": len(cls_pos), "active_reg_images": len(reg_pos)}
    return total, {
        "unsup": float(total.detach()),
        "stability": 0.0,
        "active_cls_images": len(cls_pos),
        "active_reg_images": len(reg_pos),
    }


# ------------------------------------------------------------------ 2. Dynamic GMM (Consistent-Teacher style)
def _fit_1d_two_component_gmm(
    values: list[float],
    *,
    max_iter: int = 15,
    eps: float = 1e-6,
) -> tuple[float, float, float, float, float, float]:
    """Fit a deterministic 2-component 1D GMM via EM.

    Returns ``(mu_lo, sd_lo, mu_hi, sd_hi, pi_lo, pi_hi)`` with ``mu_lo <= mu_hi``.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    q25 = xs[n // 4]
    q75 = xs[(3 * n) // 4]
    mu0, mu1 = q25, max(q75, q25 + 0.05)
    var0 = var1 = 0.03
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
        pi0, pi1 = pi1, pi0
    return mu0, math.sqrt(var0), mu1, math.sqrt(var1), pi0, pi1


def _gmm_bayes_boundary(mu0: float, sd0: float, pi0: float, mu1: float, sd1: float, pi1: float) -> float:
    """Find x in [mu0, mu1] where pi1 * N(x; mu1, sd1^2) == pi0 * N(x; mu0, sd0^2)."""
    if mu1 <= mu0 + 1e-4:
        return mu1

    def _log_diff(x: float) -> float:
        s0 = max(sd0, 1e-3)
        s1 = max(sd1, 1e-3)
        lp1 = math.log(max(pi1, 1e-6)) - math.log(s1) - 0.5 * ((x - mu1) / s1) ** 2
        lp0 = math.log(max(pi0, 1e-6)) - math.log(s0) - 0.5 * ((x - mu0) / s0) ** 2
        return lp1 - lp0

    lo, hi = mu0, mu1
    if _log_diff(lo) >= 0.0:
        return mu1 - 0.35 * sd1
    if _log_diff(hi) <= 0.0:
        return mu1
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _log_diff(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DynamicGMMThreshold:
    """Consistent-Teacher-style per-class 1D GMM adaptive threshold (checkpoint-safe)."""

    def __init__(
        self,
        num_foreground_classes: int = 10,
        *,
        base_threshold: float = 0.70,
        buffer_size: int = 256,
        min_samples: int = 20,
        min_score_input: float = 0.15,
        min_threshold: float = 0.50,
        max_threshold: float = 0.90,
    ):
        self.num_classes = int(num_foreground_classes)
        self.base_threshold = float(base_threshold)
        self.buffer_size = int(buffer_size)
        self.min_samples = int(min_samples)
        self.min_score_input = float(min_score_input)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.buffers: dict[int, list[float]] = {c: [] for c in range(1, self.num_classes + 1)}
        self.thresholds: dict[int, float] = {c: float(base_threshold) for c in range(1, self.num_classes + 1)}

    def update_and_fit(self, labels: list[int], scores: list[float]) -> None:
        touched: set[int] = set()
        for cls_id, sc in zip(labels, scores):
            cls_id = int(cls_id)
            if cls_id in self.buffers and self.min_score_input <= float(sc) <= 1.0:
                buf = self.buffers[cls_id]
                buf.append(float(sc))
                if len(buf) > self.buffer_size:
                    del buf[: len(buf) - self.buffer_size]
                touched.add(cls_id)
        for cls_id in touched:
            buf = self.buffers[cls_id]
            if len(buf) >= self.min_samples:
                mu_lo, sd_lo, mu_hi, sd_hi, pi_lo, pi_hi = _fit_1d_two_component_gmm(buf)
                # Use the stricter of Bayes posterior boundary and high-component lower shoulder
                # so the threshold never collapses into the background noise tail.
                boundary = _gmm_bayes_boundary(mu_lo, sd_lo, pi_lo, mu_hi, sd_hi, pi_hi)
                tau = max(boundary, mu_hi - 0.35 * sd_hi)
                tau = max(self.min_threshold, min(self.max_threshold, tau))
                self.thresholds[cls_id] = round(float(tau), 4)

    def threshold_for(self, cls_id: int) -> float:
        return float(self.thresholds.get(int(cls_id), self.base_threshold))

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "base_threshold": self.base_threshold,
            "buffer_size": self.buffer_size,
            "min_samples": self.min_samples,
            "min_score_input": self.min_score_input,
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
