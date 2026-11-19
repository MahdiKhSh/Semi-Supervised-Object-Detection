"""CPU gradient checks for the actual differentiable consistency terms."""
import pytest

torch = pytest.importorskip("torch")
from ssod_research.consistency import weighted_consistency_loss


def test_consistency_backpropagates():
    logits = torch.tensor([[0.8, 0.4, -0.2], [0.1, 0.7, 0.0]], requires_grad=True)
    pred = torch.tensor([[2., 3., 7., 9.], [3., 2., 10., 8.]], requires_grad=True)
    target = torch.tensor([[2., 2., 8., 9.], [3., 1., 10., 9.]])
    terms = weighted_consistency_loss(
        logits, pred, target, torch.tensor([0, 1]),
        torch.tensor([0.9, 0.8]), torch.tensor([0.03, 0.1]))
    total = sum(terms.values())
    assert total.isfinite().item()
    total.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_empty_instances_no_nan():
    logits = torch.empty((0, 3), requires_grad=True)
    boxes = torch.empty((0, 4), requires_grad=True)
    out = weighted_consistency_loss(
        logits, boxes, boxes, torch.empty(0, dtype=torch.long),
        torch.empty(0), torch.empty(0))
    assert all(v.item() == 0 for v in out.values())


def test_individual_ablation_weights():
    logits = torch.tensor([[0., 1., -2.]], requires_grad=True)
    box = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
    out = weighted_consistency_loss(
        logits, box, torch.tensor([[0., 0., 2., 2.]]),
        torch.tensor([1]), torch.tensor([0.8]), torch.tensor([0.1]),
        box_weight=0, class_weight=0, confidence_weight=1)
    assert out["loss_stability_box"].item() == 0
    assert out["loss_stability_class"].item() == 0
    assert out["loss_stability_confidence"].item() >= 0


def test_invalid_references_are_ignored():
    logits = torch.tensor([[1., 1., 0.]], requires_grad=True)
    box = torch.tensor([[1., 1., 2., 2.]], requires_grad=True)
    result = weighted_consistency_loss(
        logits, box, torch.tensor([[float("nan"), 0., 1., 1.]]),
        torch.tensor([0]), torch.tensor([0.8]), torch.tensor([0.1]))
    assert all(v.isfinite().item() for v in result.values())
