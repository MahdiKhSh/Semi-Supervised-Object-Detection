"""Exercise the *actual* TorchVision ROI consistency branch, not only the loss."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from ssod_research.native_training import build_model, consistency_roi_loss


def test_native_roi_consistency_has_student_gradients():
    model = build_model(pretrained_backbone=False, max_side=128)
    model.train()
    image = torch.ones(3, 96, 128, dtype=torch.float32) * .4
    target = {
        "boxes": torch.tensor([[24., 14., 86., 68.]]),
        "labels": torch.tensor([3], dtype=torch.long),
        "scores": torch.tensor([.92]),
        "uncertainty": torch.tensor([.05]),
    }
    loss_components = consistency_roi_loss(
        model, [image], [target],
        box_weight=1., class_weight=1., confidence_weight=.25,
        reliability_scale=4.)
    assert loss_components is not None
    assert set(loss_components) == {
        "loss_stability_box", "loss_stability_class", "loss_stability_confidence"}
    total = sum(loss_components.values())
    assert torch.isfinite(total).item()
    total.backward()
    assert model.roi_heads.box_predictor.cls_score.weight.grad is not None
    assert model.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0
    assert model.roi_heads.box_predictor.bbox_pred.weight.grad is not None


def test_consistency_empty_boxes_is_skipped():
    model = build_model(pretrained_backbone=False, max_side=128)
    model.train()
    target = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty(0, dtype=torch.long),
        "scores": torch.empty(0),
        "uncertainty": torch.empty(0),
    }
    result = consistency_roi_loss(
        model, [torch.zeros(3, 64, 128)], [target],
        box_weight=1., class_weight=1., confidence_weight=1.,
        reliability_scale=4.)
    assert result is None
