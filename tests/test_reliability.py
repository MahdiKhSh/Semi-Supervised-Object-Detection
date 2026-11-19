import numpy as np
import pytest
from ssod_research.reliability import ReliabilityConfig, select_pseudo_labels


def test_uncertainty_filters_when_enabled():
    cfg = ReliabilityConfig(min_score=0.5, localization_scale=4.0, min_quality=0.6)
    keep, quality, loc = select_pseudo_labels([0.9, 0.9], [0, 1], [0.01, 0.5], cfg)
    assert keep.tolist() == [True, False]
    assert quality[0] > quality[1] and loc[0] > loc[1]


def test_nan_is_rejected():
    keep, quality, _ = select_pseudo_labels([float("nan"), 0.8], [0, 1], [0.0, float("inf")])
    assert keep.tolist() == [False, False]
    np.testing.assert_array_equal(quality, [0, 0])


def test_class_threshold_and_mismatch():
    cfg = ReliabilityConfig(use_class_thresholds=True, class_thresholds=(0.5, 0.9))
    keep, _, _ = select_pseudo_labels([0.6, 0.6], [0, 1], config=cfg)
    assert keep.tolist() == [True, False]
    with pytest.raises(ValueError):
        select_pseudo_labels([0.1], [2], config=cfg)
    with pytest.raises(ValueError):
        select_pseudo_labels([0.9, 0.9], [0], config=cfg)
