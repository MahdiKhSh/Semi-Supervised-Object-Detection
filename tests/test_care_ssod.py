"""Unit and integration tests for CaRe-SSOD and the traffic-sign class-10 bug fix."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def test_consistency_traffic_sign_class_10_bug_fixed():
    """Regression test: TorchVision class 10 (traffic sign) must NOT crash consistency loss."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.native_training import BDD_CLASSES, build_model, consistency_roi_loss

    assert BDD_CLASSES[10] == "traffic sign"
    model = build_model(pretrained_backbone=False, max_side=128)
    model.train()
    image = torch.ones(3, 96, 128, dtype=torch.float32) * 0.4
    target = {
        "boxes": torch.tensor([[20.0, 15.0, 60.0, 55.0]]),
        "labels": torch.tensor([10], dtype=torch.long),  # traffic sign = class 10!
        "scores": torch.tensor([0.91]),
        "uncertainty": torch.tensor([0.04]),
    }
    losses = consistency_roi_loss(
        model, [image], [target],
        box_weight=1.0, class_weight=1.0, confidence_weight=0.25,
        reliability_scale=4.0,
    )
    assert losses is not None
    total = sum(losses.values())
    assert torch.isfinite(total).item()
    total.backward()
    assert model.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0


def test_visibility_degradation_separates_night_fog_from_daytime():
    torch = pytest.importorskip("torch")
    from ssod_research.care_ssod import estimate_visibility_degradation

    # Bright, high-contrast daytime synthetic scene
    day = torch.zeros(3, 64, 64)
    day[:, :32, :] = 0.85
    day[:, 32:, :] = 0.25
    res_day = estimate_visibility_degradation(day)

    # Dark low-contrast nighttime scene with headlight glare spot
    night = torch.full((3, 64, 64), 0.06)
    night[:, 28:34, 28:34] = 0.99
    res_night = estimate_visibility_degradation(night)

    # Low-contrast foggy scene
    fog = torch.full((3, 64, 64), 0.50) + 0.01 * torch.randn(3, 64, 64)
    res_fog = estimate_visibility_degradation(fog.clamp(0, 1))

    assert res_day["regime"] == "clear"
    assert res_night["regime"] == "degraded"
    assert res_night["degradation"] > res_day["degradation"]
    assert res_fog["low_contrast"] > res_day["low_contrast"]


def test_nwd_is_smoother_than_iou_on_small_objects():
    torch = pytest.importorskip("torch")
    from torchvision.ops import box_iou
    from ssod_research.care_ssod import pairwise_nwd

    # Tiny 12x12 traffic light shifted by 2px in x and y
    a = torch.tensor([[50.0, 50.0, 62.0, 62.0]])
    b = torch.tensor([[52.0, 52.0, 64.0, 64.0]])
    iou = float(box_iou(a, b)[0, 0])
    nwd = float(pairwise_nwd(a, b)[0, 0])
    assert iou < 0.60
    assert nwd > 0.78
    assert nwd > iou


def test_online_calibrator_adapts_and_serializes():
    from ssod_research.care_ssod import OnlineVisibilityClassCalibrator

    cal = OnlineVisibilityClassCalibrator(num_foreground_classes=10, base_threshold=0.7, momentum=0.5)
    # Class 3 (car) has high confidence in clear daytime; class 2 (rider) has lower confidence in degraded night
    for _ in range(10):
        cal.update([3, 3], [0.90, 0.88], "clear")
        cal.update([2], [0.48], "degraded")

    tau_car_clear, low_car = cal.thresholds(3, "clear")
    tau_rider_night, low_rider = cal.thresholds(2, "degraded")
    assert tau_rider_night < tau_car_clear
    assert low_rider < tau_rider_night
    assert low_car < tau_car_clear

    state = cal.state_dict()
    cal2 = OnlineVisibilityClassCalibrator(num_foreground_classes=10, base_threshold=0.7, momentum=0.5)
    cal2.load_state_dict(state)
    assert cal2.thresholds(2, "degraded") == pytest.approx(cal.thresholds(2, "degraded"))
    assert cal2.updates == cal.updates


def test_ambiguity_zone_background_suppression():
    torch = pytest.importorskip("torch")
    from ssod_research.care_ssod import filter_ambiguous_background_proposals

    props = [torch.tensor([
        [10.0, 10.0, 50.0, 50.0],    # overlaps Zone-1 positive -> MUST KEEP
        [100.0, 100.0, 140.0, 140.0], # overlaps Zone-2 ambiguous -> MUST SUPPRESS
        [200.0, 200.0, 240.0, 240.0], # genuine background -> MUST KEEP
    ])]
    pos = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
    ambig = [torch.tensor([[102.0, 102.0, 138.0, 138.0]])]

    filtered, suppressed = filter_ambiguous_background_proposals(props, pos, ambig)
    assert suppressed == 1
    assert len(filtered[0]) == 2
    assert torch.allclose(filtered[0][0], props[0][0])
    assert torch.allclose(filtered[0][1], props[0][2])


def test_care_unsupervised_loss_handles_zone1_and_zone2_only_batches():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.care_ssod import care_unsupervised_loss, strong_view_care
    from ssod_research.native_training import build_model, target_for_training

    model = build_model(pretrained_backbone=False, max_side=128)
    model.train()
    img = torch.rand(3, 96, 128)

    # Case A: ONLY Zone-2 ambiguous detections (zero Zone-1 positives!) -> must STILL train softly!
    empty_z1 = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "scores": torch.zeros((0,)),
        "probs": torch.zeros((0, 11)),
        "reliability": torch.zeros((0,)),
    }
    probs_z2 = torch.full((1, 11), 0.02)
    probs_z2[0, 10] = 0.80  # traffic sign = class 10!
    z2 = {
        "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
        "labels": torch.tensor([10], dtype=torch.long),
        "scores": torch.tensor([0.55]),
        "probs": probs_z2,
        "reliability": torch.tensor([0.45]),
    }
    a_img, a_z1, a_z2 = strong_view_care(img, empty_z1, z2, rng=random.Random(7), enable_cutout=True)
    loss, stats = care_unsupervised_loss(
        model, [a_img], [target_for_training(a_z1, image_id=1)], [a_z1], [a_z2],
    )
    assert loss is not None and torch.isfinite(loss).item()
    assert stats["stability"] > 0.0
    loss.backward()
    assert model.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0


def test_care_ssod_deterministic_interrupt_resume_and_preset(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from make_synthetic_bdd import create
    from prepare_bdd100k import main as prepare_main
    from run_suite import generate_jobs, parse_args as suite_parse
    from ssod_research.native_training import parse_args, train
    from ssod_research.recovery import read_json, safe_torch_load

    # Verify --preset care matrix generation
    sargs = suite_parse(["--preset", "care", "--output", str(tmp_path / "care_plan")])
    jobs = generate_jobs(sargs)
    assert any(j["name"] == "05pct_s42/care_ssod" for j in jobs)
    assert any(j["name"] == "05pct_s42/care_no_trizone" for j in jobs)
    assert not any(j["name"] == "05pct_s123/care_no_trizone" for j in jobs)
    assert any(j["name"] == "10pct_s2026/care_ssod" for j in jobs)

    # Deterministic interrupt-and-resume check for care_ssod
    src = tmp_path / "src"
    paths = create(src, n_train=10, n_val=4, seed=42)
    split_dir = tmp_path / "split"
    prepare_main([
        "--train-labels", paths["train_labels"],
        "--val-labels", paths["val_labels"],
        "--output", str(split_dir),
        "--fraction", "0.4",
        "--seed", "42",
    ])

    def _argv(out_dir: Path, stop_after: int = 0):
        return [
            "--data-root", str(split_dir),
            "--train-images", paths["train_images"],
            "--val-images", paths["val_images"],
            "--output", str(out_dir),
            "--variant", "care_ssod",
            "--seed", "42",
            "--steps", "2",
            "--batch-size", "1",
            "--max-side", "96",
            "--save-every", "1",
            "--eval-every", "2",
            "--eval-max-images", "2",
            "--device", "cpu",
            "--amp", "off",
            "--no-pretrained-backbone",
            "--deterministic",
            "--stop-after-steps", str(stop_after),
        ]

    out_full = tmp_path / "care_full"
    out_res = tmp_path / "care_res"

    train(parse_args(_argv(out_full, stop_after=0)))
    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_argv(out_res, stop_after=1)))

    meta = train(parse_args(_argv(out_res, stop_after=0)))
    assert meta["status"] == "completed"
    assert meta["resume_mode"] == "exact"

    st_a = safe_torch_load(out_full / "checkpoints" / "ckpt_000000002.pth")
    st_b = safe_torch_load(out_res / "checkpoints" / "ckpt_000000002.pth")
    assert st_a["algorithm_state"]["care_calibrator"] == st_b["algorithm_state"]["care_calibrator"]
    for key in st_a["student"]:
        a, b = st_a["student"][key], st_b["student"][key]
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        else:
            assert torch.equal(a, b)
