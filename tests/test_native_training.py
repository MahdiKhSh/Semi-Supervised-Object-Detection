import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from ssod_research.native_training import (
    build_model, compute_threshold, ema_update, sample_indices, BDD_CLASSES,
    target_for_training,
)


def test_deterministic_step_sampling():
    assert sample_indices(15, 8, 42, 3, 1) == sample_indices(15, 8, 42, 3, 1)
    assert sample_indices(15, 8, 42, 3, 0) != sample_indices(15, 8, 42, 3, 1)


def test_adaptive_and_condition_thresholds():
    counts = {1: 10, 2: 1000}
    common = compute_threshold(2, .7, adaptive=True, counts=counts, strength=.15,
                               condition_enabled=False, attrs={}, condition_offset=.05)
    rare = compute_threshold(1, .7, adaptive=True, counts=counts, strength=.15,
                             condition_enabled=False, attrs={}, condition_offset=.05)
    assert rare < common
    difficult = compute_threshold(1, .7, adaptive=False, counts=counts, strength=.15,
                                  condition_enabled=True,
                                  attrs={"timeofday": "night"}, condition_offset=.05)
    assert difficult == pytest.approx(.75)


def test_ema():
    student = torch.nn.Linear(2, 1)
    teacher = torch.nn.Linear(2, 1)
    with torch.no_grad():
        student.weight.fill_(2)
        teacher.weight.fill_(0)
    ema_update(teacher, student, .5)
    assert torch.allclose(teacher.weight, torch.ones_like(teacher.weight))


def test_pseudo_to_native_targets():
    p = {"boxes": torch.tensor([[1., 1., 4., 5.]]),
         "labels": torch.tensor([3]), "scores": torch.tensor([.9]),
         "uncertainty": torch.tensor([.01])}
    t = target_for_training(p, image_id=11)
    assert t["area"].tolist() == [12.0]
    assert t["image_id"].item() == 11


def test_model_build_smoke():
    m = build_model(pretrained_backbone=False, max_side=128)
    assert m.roi_heads.box_predictor.cls_score.out_features == len(BDD_CLASSES)
