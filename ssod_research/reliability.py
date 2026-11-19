"""Transparent research prototype for pseudo-label selection.

Does NOT by itself establish paper novelty or implement full consistency training.
Separate score and localization reliability. This module is numpy-only for tests.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReliabilityConfig:
    min_score: float = 0.5
    localization_scale: float = 4.0
    min_quality: float = 0.0
    class_thresholds: tuple[float, ...] = ()
    use_localization: bool = True
    use_class_thresholds: bool = False

    def __post_init__(self):
        if not (0 <= self.min_score <= 1 and 0 <= self.min_quality <= 1):
            raise ValueError("thresholds must be in [0,1]")
        if self.localization_scale < 0:
            raise ValueError("localization_scale must be nonnegative")
        if any(not 0 <= x <= 1 for x in self.class_thresholds):
            raise ValueError("class thresholds must be in [0,1]")


def select_pseudo_labels(scores, classes, localization_uncertainty=None, config=None):
    """Return selected mask, calibrated quality, and component reliability.

    A zero uncertainty implies no localization penalty. NaN/inf values are rejected.
    Class thresholds are fixed experimental values, not learned calibration.
    """
    cfg = config or ReliabilityConfig()
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(classes, dtype=np.int64).reshape(-1)
    if len(score) != len(label):
        raise ValueError("score/label lengths differ")
    if localization_uncertainty is None:
        uncertainty = np.zeros_like(score)
    else:
        uncertainty = np.asarray(localization_uncertainty, dtype=np.float64).reshape(-1)
        if len(uncertainty) != len(score):
            raise ValueError("uncertainty/score lengths differ")
    finite = np.isfinite(score) & np.isfinite(uncertainty)
    valid = finite & (score >= 0) & (score <= 1) & (uncertainty >= 0) & (label >= 0)
    safe_scores = np.where(valid, score, 0)
    safe_unc = np.where(valid, uncertainty, np.inf)
    loc = np.exp(-cfg.localization_scale * safe_unc) if cfg.use_localization else np.ones_like(score)
    quality = safe_scores * loc
    thresholds = np.full(len(label), cfg.min_score, dtype=np.float64)
    if cfg.use_class_thresholds:
        if len(cfg.class_thresholds) == 0:
            raise ValueError("class_thresholds required when enabled")
        if np.any(label[valid] >= len(cfg.class_thresholds)):
            raise ValueError("class index exceeds threshold table")
        thresholds[valid] = np.asarray(cfg.class_thresholds)[label[valid]]
    selected = valid & (safe_scores >= thresholds) & (quality >= cfg.min_quality)
    quality[~valid] = 0
    return selected, quality, loc
