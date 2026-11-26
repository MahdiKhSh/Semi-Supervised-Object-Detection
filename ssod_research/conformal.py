"""Weighted Conformal Risk Control (CRC) for Photometric-Shift SSOD (CoRe-SSOD).

Replaces ad-hoc pseudo-label confidence heuristics with a mathematically
principled finite-sample risk control procedure (Angelopoulos et al., ICLR 2024;
Tibshirani et al., NeurIPS 2019) using a strictly held-out calibration fold
carved from the labeled training split (zero validation leakage).

For each visibility regime r in {"clear", "degraded"} (estimated strictly from
unlabeled RGB pixels without weather/time metadata) and risk budget alpha:

    L_tilde(x_i; lam) = sup_{lam' >= lam} FDR(x_i; lam') in [0, 1]
    R_hat_r(lam)      = sum_i w_r(d(x_i)) * L_tilde(x_i; lam) / sum_i w_r(d(x_i))
    U_r(lam)          = (n_eff / (n_eff + 1)) * R_hat_r(lam) + 1 / (n_eff + 1)
    lam_hat_r(alpha)  = inf { lam in Grid : U_r(lam) <= alpha }

Tri-Zone thresholds follow automatically from two interpretable risk levels:
    tau_high = lam_hat(alpha_strict)   (Zone 1: positive + GIoU + KL)
    tau_low  = lam_hat(alpha_loose)    (Zone 2: bg-suppressed + soft KL only)
"""
from __future__ import annotations

import math
import random
from typing import Any

import torch
from torchvision.ops import box_iou

from ssod_research.care_ssod import (
    _teacher_detect_with_probs,
    estimate_visibility_degradation,
    jensen_shannon_divergence,
    pairwise_nwd,
)


def split_calibration_indices(
    dataset_size: int,
    *,
    seed: int = 42,
    cal_fraction: float = 0.08,
    max_cal: int = 256,
) -> tuple[list[int], list[int]]:
    """Deterministically partition labeled training indices into (train_ids, cal_ids).

    Strictly disjoint: calibration images are NEVER used for supervised student
    gradient updates, preserving exchangeability for the Conformal Risk Control
    guarantee while NEVER touching the validation set.
    """
    n = int(dataset_size)
    if n < 2:
        return list(range(n)), []
    indices = list(range(n))
    rng = random.Random(int(seed) * 9973 + 17)
    rng.shuffle(indices)
    n_cal = max(1, min(int(max_cal), n - 1, int(round(n * float(cal_fraction)))))
    cal_ids = sorted(indices[:n_cal])
    train_ids = sorted(indices[n_cal:])
    assert set(cal_ids).isdisjoint(set(train_ids))
    return train_ids, cal_ids


def _match_detections_to_gt(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    iou_thresh: float = 0.50,
) -> list[int]:
    """Greedy class-matched 1:1 TP/FP assignment sorted by descending score."""
    n_pred = len(pred_boxes)
    if n_pred == 0:
        return []
    tp = [0] * n_pred
    if len(gt_boxes) == 0:
        return tp
    iou_mat = box_iou(pred_boxes.float(), gt_boxes.float()).detach().cpu().numpy()
    p_lbl = pred_labels.detach().cpu().tolist()
    g_lbl = gt_labels.detach().cpu().tolist()
    order = sorted(range(n_pred), key=lambda idx: float(pred_scores[idx].item()), reverse=True)
    matched_gt: set[int] = set()
    for p_idx in order:
        best_iou = iou_thresh - 1e-9
        best_g = -1
        for g_idx, g_c in enumerate(g_lbl):
            if g_idx in matched_gt or g_c != p_lbl[p_idx]:
                continue
            val = float(iou_mat[p_idx, g_idx])
            if val >= best_iou:
                best_iou = val
                best_g = g_idx
        if best_g >= 0:
            tp[p_idx] = 1
            matched_gt.add(best_g)
    return tp


def _monotonized_fdr_curve(
    reliabilities: list[float],
    tp_flags: list[int],
    grid: list[float],
) -> tuple[list[float], list[int]]:
    """Compute bounded monotone upper-envelope FDR curve L_tilde(x; lam) in [0, 1].

    L(x; lam) = FP(lam) / max(1, N_accepted(lam)), with L = 0 when N_accepted = 0.
    L_tilde(x; lam) = max_{lam' >= lam} L(x; lam') is provably non-increasing in lam.
    """
    raw_fdr = []
    accepted_counts = []
    for lam in grid:
        n_acc = 0
        n_fp = 0
        for q, is_tp in zip(reliabilities, tp_flags):
            if q >= lam:
                n_acc += 1
                if not is_tp:
                    n_fp += 1
        raw_fdr.append((n_fp / n_acc) if n_acc > 0 else 0.0)
        accepted_counts.append(n_acc)

    # Suffix maximum from high lam -> low lam guarantees non-increasing L_tilde(lam)
    mono = [0.0] * len(grid)
    running_max = 0.0
    for idx in range(len(grid) - 1, -1, -1):
        if raw_fdr[idx] > running_max:
            running_max = raw_fdr[idx]
        mono[idx] = running_max
    return mono, accepted_counts


def solve_conformal_threshold(
    loss_curves: list[list[float]],
    weights: list[float],
    grid: list[float],
    alpha: float,
    *,
    min_threshold: float = 0.25,
    max_threshold: float = 0.90,
) -> dict[str, Any]:
    """Solve weighted Conformal Risk Control for the smallest valid threshold."""
    if not loss_curves or not weights:
        return {
            "threshold": max(min_threshold, min(max_threshold, 0.70)),
            "empirical_risk": None,
            "conformal_upper": None,
            "n_eff": 0.0,
            "bound_satisfied": False,
        }
    w_sum = sum(weights)
    w_sq_sum = sum(w * w for w in weights)
    if w_sum <= 0.0 or w_sq_sum <= 0.0:
        return {
            "threshold": max(min_threshold, min(max_threshold, 0.70)),
            "empirical_risk": None,
            "conformal_upper": None,
            "n_eff": 0.0,
            "bound_satisfied": False,
        }
    n_eff = (w_sum * w_sum) / w_sq_sum

    chosen_idx = None
    best_emp = 1.0
    best_upper = 1.0
    for g_idx, lam in enumerate(grid):
        if lam < min_threshold or lam > max_threshold:
            continue
        r_hat = sum(w * curve[g_idx] for w, curve in zip(weights, loss_curves)) / w_sum
        upper = (n_eff / (n_eff + 1.0)) * r_hat + (1.0 / (n_eff + 1.0))
        if upper <= float(alpha):
            chosen_idx = g_idx
            best_emp = r_hat
            best_upper = upper
            break

    if chosen_idx is not None:
        return {
            "threshold": float(grid[chosen_idx]),
            "empirical_risk": round(float(best_emp), 5),
            "conformal_upper": round(float(best_upper), 5),
            "n_eff": round(float(n_eff), 2),
            "bound_satisfied": True,
        }

    # No grid threshold below max_threshold met alpha; return strictest safe ceiling honestly.
    valid_indices = [i for i, lam in enumerate(grid) if min_threshold <= lam <= max_threshold]
    last_idx = valid_indices[-1] if valid_indices else len(grid) - 1
    r_hat = sum(w * curve[last_idx] for w, curve in zip(weights, loss_curves)) / w_sum
    upper = (n_eff / (n_eff + 1.0)) * r_hat + (1.0 / (n_eff + 1.0))
    return {
        "threshold": float(max_threshold),
        "empirical_risk": round(float(r_hat), 5),
        "conformal_upper": round(float(upper), 5),
        "n_eff": round(float(n_eff), 2),
        "bound_satisfied": False,
    }


class WeightedConformalRiskCalibrator:
    """Online, checkpoint-serializable Weighted Conformal Risk Controller.

    Tracks the unlabeled photometric degradation distribution d(x) across 5 bins
    in [0, 1], computes density-ratio importance weights for clear and degraded
    regimes, and periodically solves exact Conformal Risk Control on the held-out
    labeled calibration split to yield (tau_high, tau_low) per class & regime.
    """

    def __init__(
        self,
        num_foreground_classes: int = 10,
        *,
        alpha_strict: float = 0.15,
        alpha_loose: float = 0.40,
        min_high: float = 0.35,
        max_high: float = 0.90,
        min_low: float = 0.18,
        shrinkage_prior: float = 15.0,
        use_visibility_weighting: bool = True,
        num_vis_bins: int = 5,
        iou_match: float = 0.50,
    ):
        self.num_classes = int(num_foreground_classes)
        self.alpha_strict = float(alpha_strict)
        self.alpha_loose = float(alpha_loose)
        self.min_high = float(min_high)
        self.max_high = float(max_high)
        self.min_low = float(min_low)
        self.shrinkage_prior = float(shrinkage_prior)
        self.use_visibility_weighting = bool(use_visibility_weighting)
        self.num_vis_bins = int(num_vis_bins)
        self.iou_match = float(iou_match)
        self.grid = [round(0.05 + 0.02 * k, 4) for k in range(48)]  # 0.05 .. 0.99

        # Online empirical histogram of unlabeled RGB degradation scores d(x)
        self.unlabeled_vis_hist: list[float] = [1.0] * self.num_vis_bins
        self.unlabeled_degraded_count: int = 0
        self.unlabeled_total_count: int = 0

        # Active thresholds per regime and per foreground class 1..C
        self.regime_high: dict[str, float] = {"clear": 0.70, "degraded": 0.65}
        self.regime_low: dict[str, float] = {"clear": 0.35, "degraded": 0.32}
        self.class_high: dict[str, dict[int, float]] = {
            "clear": {c: 0.70 for c in range(1, self.num_classes + 1)},
            "degraded": {c: 0.65 for c in range(1, self.num_classes + 1)},
        }
        self.class_low: dict[str, dict[int, float]] = {
            "clear": {c: 0.35 for c in range(1, self.num_classes + 1)},
            "degraded": {c: 0.32 for c in range(1, self.num_classes + 1)},
        }
        self.calibration_count: int = 0
        self.last_diagnostics: dict[str, Any] = {}

    def _vis_bin(self, degradation: float) -> int:
        idx = int(float(degradation) * self.num_vis_bins)
        return max(0, min(self.num_vis_bins - 1, idx))

    def observe_unlabeled_visibility(self, degradation: float, regime: str) -> None:
        """Record label-free RGB degradation score from unlabeled images."""
        self.unlabeled_vis_hist[self._vis_bin(degradation)] += 1.0
        self.unlabeled_total_count += 1
        if regime == "degraded":
            self.unlabeled_degraded_count += 1

    def thresholds(self, cls_id: int, regime: str, *, use_visibility: bool = True) -> tuple[float, float]:
        """Return conformal (tau_high, tau_low) for class ``cls_id`` under ``regime``."""
        reg = regime if (use_visibility and regime in ("clear", "degraded")) else "clear"
        tau_hi = float(self.class_high[reg].get(int(cls_id), self.regime_high[reg]))
        tau_lo = float(self.class_low[reg].get(int(cls_id), self.regime_low[reg]))
        tau_hi = max(self.min_high, min(self.max_high, tau_hi))
        tau_lo = max(self.min_low, min(tau_hi - 0.05, tau_lo))
        return tau_hi, tau_lo

    def update(self, labels: list[int], scores: list[float], regime: str) -> None:
        """No-op compatibility hook for shared extraction pipeline; CRC calibrates on D_cal."""
        return None

    def _importance_weights(
        self,
        cal_degradations: list[float],
        regime: str,
    ) -> list[float]:
        """Density-ratio weights w_r(d(x_i)) = p_target_r(bin) / p_cal(bin)."""
        n = len(cal_degradations)
        if n == 0:
            return []
        if not self.use_visibility_weighting:
            return [1.0] * n

        cal_hist = [1.0] * self.num_vis_bins
        for d in cal_degradations:
            cal_hist[self._vis_bin(d)] += 1.0
        cal_total = sum(cal_hist)
        p_cal = [c / cal_total for c in cal_hist]

        # Build regime-conditioned target density from the unlabeled visibility histogram
        split_bin = self._vis_bin(0.35)
        target_hist = list(self.unlabeled_vis_hist)
        if regime == "degraded":
            # Upweight degraded bins while keeping smooth support across all bins
            target_hist = [
                (v * 3.0 if b >= split_bin else v * 0.25)
                for b, v in enumerate(target_hist)
            ]
        else:
            target_hist = [
                (v * 3.0 if b < split_bin else v * 0.25)
                for b, v in enumerate(target_hist)
            ]
        t_total = sum(target_hist)
        p_target = [v / max(1e-9, t_total) for v in target_hist]

        weights = []
        for d in cal_degradations:
            b = self._vis_bin(d)
            ratio = p_target[b] / max(1e-6, p_cal[b])
            weights.append(max(0.15, min(5.0, ratio)))
        return weights

    @torch.no_grad()
    def calibrate_on_dataset(
        self,
        teacher,
        dataset,
        cal_indices: list[int],
        device: torch.device,
        *,
        step: int = 0,
        max_samples: int = 256,
        use_nwd: bool = True,
        nwd_weight: float = 0.65,
        nwd_constant: float = 12.8,
        loc_alpha: float = 2.5,
        cls_alpha: float = 1.5,
        match_floor: float = 0.40,
    ) -> dict[str, Any]:
        """Run exact Weighted Conformal Risk Control on held-out D_cal."""
        chosen = cal_indices[:max_samples]
        cal_degradations: list[float] = []
        image_curves: list[list[float]] = []
        per_class_curves: dict[int, list[tuple[list[float], float]]] = {
            c: [] for c in range(1, self.num_classes + 1)
        }
        gt_by_regime = {"clear": 0, "degraded": 0}
        all_det_records: list[tuple[float, int, str]] = []

        teacher.eval()
        for idx in chosen:
            img, target = dataset[idx]
            img_dev = img.to(device)
            vis = estimate_visibility_degradation(img_dev)
            deg = float(vis["degradation"])
            regime_i = str(vis["regime"])
            cal_degradations.append(deg)

            std = _teacher_detect_with_probs(teacher, [img_dev])[0]
            rep = _teacher_detect_with_probs(teacher, [torch.flip(img_dev, dims=(-1,))])[0]

            boxes = std["boxes"]
            labels = std["labels"]
            scores = std["scores"]
            probs = std["probs"]

            b_boxes = rep["boxes"].clone()
            width = img_dev.shape[-1]
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
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            gt_by_regime[regime_i] += len(gt_boxes)

            tp_flags = _match_detections_to_gt(
                boxes, labels, reliability, gt_boxes, gt_labels, iou_thresh=self.iou_match
            )
            rel_list = reliability.detach().cpu().tolist()
            lbl_list = labels.detach().cpu().tolist()
            for q_val, is_tp in zip(rel_list, tp_flags):
                if math.isfinite(q_val):
                    all_det_records.append((float(q_val), int(is_tp), regime_i))

            curve, _ = _monotonized_fdr_curve(rel_list, tp_flags, self.grid)
            image_curves.append(curve)

            # Per-class monotone FDR curves for classes present in predictions or GT of this image
            present_classes = set(int(c) for c in lbl_list) | set(int(c) for c in gt_labels.detach().cpu().tolist())
            for cls_id in present_classes:
                if cls_id not in per_class_curves:
                    continue
                c_rels = [q for q, c in zip(rel_list, lbl_list) if int(c) == cls_id]
                c_tps = [t for t, c in zip(tp_flags, lbl_list) if int(c) == cls_id]
                c_curve, _ = _monotonized_fdr_curve(c_rels, c_tps, self.grid)
                per_class_curves[cls_id].append((c_curve, deg))

        diagnostics: dict[str, Any] = {
            "step": int(step),
            "cal_images": len(chosen),
            "cal_gt_boxes": sum(gt_by_regime.values()),
            "alpha_strict": self.alpha_strict,
            "alpha_loose": self.alpha_loose,
            "unlabeled_degraded_frac": round(
                self.unlabeled_degraded_count / max(1, self.unlabeled_total_count), 4
            ),
            "regimes": {},
        }

        for regime in ("clear", "degraded"):
            weights = self._importance_weights(cal_degradations, regime)
            sol_hi = solve_conformal_threshold(
                image_curves, weights, self.grid, self.alpha_strict,
                min_threshold=self.min_high, max_threshold=self.max_high,
            )
            sol_lo = solve_conformal_threshold(
                image_curves, weights, self.grid, self.alpha_loose,
                min_threshold=self.min_low, max_threshold=max(self.min_low, sol_hi["threshold"] - 0.05),
            )
            self.regime_high[regime] = float(sol_hi["threshold"])
            self.regime_low[regime] = float(min(sol_hi["threshold"] - 0.05, sol_lo["threshold"]))

            # Empirical Bayes shrinkage per foreground class
            for cls_id in range(1, self.num_classes + 1):
                entries = per_class_curves[cls_id]
                if not entries:
                    self.class_high[regime][cls_id] = self.regime_high[regime]
                    self.class_low[regime][cls_id] = self.regime_low[regime]
                    continue
                c_curves = [e[0] for e in entries]
                c_degs = [e[1] for e in entries]
                c_weights = self._importance_weights(c_degs, regime)
                c_hi = solve_conformal_threshold(
                    c_curves, c_weights, self.grid, self.alpha_strict,
                    min_threshold=self.min_high, max_threshold=self.max_high,
                )
                c_lo = solve_conformal_threshold(
                    c_curves, c_weights, self.grid, self.alpha_loose,
                    min_threshold=self.min_low, max_threshold=max(self.min_low, c_hi["threshold"] - 0.05),
                )
                n_c = float(c_hi["n_eff"])
                blend = n_c / (n_c + self.shrinkage_prior)
                shrunk_hi = blend * float(c_hi["threshold"]) + (1.0 - blend) * self.regime_high[regime]
                shrunk_lo = blend * float(c_lo["threshold"]) + (1.0 - blend) * self.regime_low[regime]
                shrunk_hi = max(self.min_high, min(self.max_high, shrunk_hi))
                shrunk_lo = max(self.min_low, min(shrunk_hi - 0.05, shrunk_lo))
                self.class_high[regime][cls_id] = round(shrunk_hi, 4)
                self.class_low[regime][cls_id] = round(shrunk_lo, 4)

            # Compute empirical pooled recall and FDR at tau_high for diagnostic plots
            tau_h = self.regime_high[regime]
            acc_recs = [r for r in all_det_records if r[2] == regime and r[0] >= tau_h]
            emp_fdr = (
                sum(1 - r[1] for r in acc_recs) / len(acc_recs)
                if acc_recs else 0.0
            )
            emp_recall = sum(r[1] for r in acc_recs) / max(1, gt_by_regime[regime])
            diagnostics["regimes"][regime] = {
                "ground_truth_boxes": gt_by_regime[regime],
                "accepted_detections": len(acc_recs),
                "tau_high": round(self.regime_high[regime], 4),
                "tau_low": round(self.regime_low[regime], 4),
                "empirical_weighted_risk": sol_hi["empirical_risk"],
                "conformal_upper_bound": sol_hi["conformal_upper"],
                "bound_satisfied": sol_hi["bound_satisfied"],
                "n_eff": sol_hi["n_eff"],
                "pooled_empirical_fdr": round(float(emp_fdr), 4),
                "pooled_empirical_recall": round(float(emp_recall), 4),
            }

        self.calibration_count += 1
        self.last_diagnostics = diagnostics
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "alpha_strict": self.alpha_strict,
            "alpha_loose": self.alpha_loose,
            "min_high": self.min_high,
            "max_high": self.max_high,
            "min_low": self.min_low,
            "shrinkage_prior": self.shrinkage_prior,
            "use_visibility_weighting": self.use_visibility_weighting,
            "num_vis_bins": self.num_vis_bins,
            "iou_match": self.iou_match,
            "unlabeled_vis_hist": [float(v) for v in self.unlabeled_vis_hist],
            "unlabeled_degraded_count": int(self.unlabeled_degraded_count),
            "unlabeled_total_count": int(self.unlabeled_total_count),
            "regime_high": {str(k): float(v) for k, v in self.regime_high.items()},
            "regime_low": {str(k): float(v) for k, v in self.regime_low.items()},
            "class_high": {
                str(r): {int(c): float(v) for c, v in tbl.items()}
                for r, tbl in self.class_high.items()
            },
            "class_low": {
                str(r): {int(c): float(v) for c, v in tbl.items()}
                for r, tbl in self.class_low.items()
            },
            "calibration_count": int(self.calibration_count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "unlabeled_vis_hist" in state:
            self.unlabeled_vis_hist = [float(v) for v in state["unlabeled_vis_hist"]]
        self.unlabeled_degraded_count = int(state.get("unlabeled_degraded_count", 0))
        self.unlabeled_total_count = int(state.get("unlabeled_total_count", 0))
        if "regime_high" in state:
            self.regime_high = {str(k): float(v) for k, v in state["regime_high"].items()}
        if "regime_low" in state:
            self.regime_low = {str(k): float(v) for k, v in state["regime_low"].items()}
        if "class_high" in state:
            self.class_high = {
                str(r): {int(c): float(v) for c, v in tbl.items()}
                for r, tbl in state["class_high"].items()
            }
        if "class_low" in state:
            self.class_low = {
                str(r): {int(c): float(v) for c, v in tbl.items()}
                for r, tbl in state["class_low"].items()
            }
        self.calibration_count = int(state.get("calibration_count", 0))
