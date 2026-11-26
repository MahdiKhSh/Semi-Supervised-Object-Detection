"""Tests for CoRe-SSOD (Weighted Conformal Risk Control), Soft Teacher, GMM, and Q1 presets."""
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


def test_split_calibration_indices_disjoint_and_deterministic():
    from ssod_research.conformal import split_calibration_indices

    assert split_calibration_indices(1) == ([0], [])
    tr1, cal1 = split_calibration_indices(100, seed=42, cal_fraction=0.08, max_cal=256)
    tr2, cal2 = split_calibration_indices(100, seed=42, cal_fraction=0.08, max_cal=256)
    assert tr1 == tr2 and cal1 == cal2
    assert len(cal1) == 8 and len(tr1) == 92
    assert set(tr1).isdisjoint(set(cal1))
    assert sorted(tr1 + cal1) == list(range(100))


def test_monotonized_fdr_curve_and_conformal_risk_guarantee():
    from ssod_research.conformal import _monotonized_fdr_curve, solve_conformal_threshold

    grid = [round(0.1 * k, 2) for k in range(1, 10)]  # 0.1 .. 0.9
    # Low-reliability detections are false positives; high-reliability (>= 0.7) are true positives
    rels = [0.2, 0.35, 0.55, 0.72, 0.85, 0.92]
    tps = [0, 0, 0, 1, 1, 1]
    curve, _ = _monotonized_fdr_curve(rels, tps, grid)
    # Must be strictly non-increasing in lambda and bounded in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in curve)
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[-1] == 0.0  # at 0.9, only 0.92 (TP) is accepted -> FDR = 0

    # 30 identical calibration images -> finite-sample 1/(n+1) = 0.032 < 0.10
    curves = [curve for _ in range(30)]
    weights = [1.0] * 30
    sol_strict = solve_conformal_threshold(curves, weights, grid, alpha=0.10, min_threshold=0.1, max_threshold=0.9)
    sol_loose = solve_conformal_threshold(curves, weights, grid, alpha=0.45, min_threshold=0.1, max_threshold=0.9)
    assert sol_strict["bound_satisfied"] is True
    assert sol_strict["conformal_upper"] <= 0.10
    # 0.6 is the smallest grid point above the highest FP (0.55), giving FDR=0 <= 0.10
    assert sol_strict["threshold"] == pytest.approx(0.6)
    assert sol_loose["threshold"] <= sol_strict["threshold"]


def test_weighted_conformal_calibrator_serialization():
    from ssod_research.conformal import WeightedConformalRiskCalibrator

    cal = WeightedConformalRiskCalibrator(num_foreground_classes=10, alpha_strict=0.15, alpha_loose=0.40)
    cal.observe_unlabeled_visibility(0.80, "degraded")
    cal.observe_unlabeled_visibility(0.10, "clear")
    cal.regime_high["degraded"] = 0.58
    cal.class_high["degraded"][3] = 0.61
    state = cal.state_dict()

    cal2 = WeightedConformalRiskCalibrator(num_foreground_classes=10)
    cal2.load_state_dict(state)
    assert cal2.state_dict() == state
    assert cal2.thresholds(3, "degraded")[0] == pytest.approx(0.61)


def test_dynamic_gmm_threshold_separates_bimodal_scores():
    from ssod_research.baselines import DynamicGMMThreshold

    gmm = DynamicGMMThreshold(num_foreground_classes=10, base_threshold=0.70, min_samples=20)
    rng = random.Random(42)
    low_scores = [rng.uniform(0.10, 0.28) for _ in range(40)]
    high_scores = [rng.uniform(0.76, 0.95) for _ in range(40)]
    scores = low_scores + high_scores
    labels = [3] * len(scores)
    gmm.update_and_fit(labels, scores)

    tau = gmm.threshold_for(3)
    assert 0.65 <= tau <= 0.90

    st = gmm.state_dict()
    gmm2 = DynamicGMMThreshold(num_foreground_classes=10)
    gmm2.load_state_dict(st)
    assert gmm2.threshold_for(3) == pytest.approx(tau)


def test_soft_teacher_extract_and_soft_negative_loss_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.baselines import soft_teacher_extract, soft_teacher_unsupervised_loss
    from ssod_research.native_training import build_model, target_for_training

    student = build_model(pretrained_backbone=False, max_side=128)
    teacher = build_model(pretrained_backbone=False, max_side=128)
    student.train()
    teacher.eval()
    img = torch.rand(3, 96, 128)

    # Extract with low cls_threshold so untrained random teacher produces candidates
    pseudo, stats, cache = soft_teacher_extract(
        teacher, [img, img], cls_threshold=0.01, reg_threshold=0.01, jitter_times=2, jitter_thresh=1.0, rng=random.Random(3),
    )
    assert stats["candidates"] >= 0
    if len(pseudo[0]["boxes"]) == 0:
        # Force one synthetic pseudo-box to guarantee loss + soft-negative weighting path runs
        pseudo[0] = {
            "boxes": torch.tensor([[16.0, 16.0, 64.0, 64.0]]),
            "labels": torch.tensor([3], dtype=torch.long),
            "scores": torch.tensor([0.85]),
            "reg_valid": torch.tensor([True]),
            "jitter_std": torch.tensor([0.01]),
        }
    target = target_for_training(pseudo[0], image_id=1)
    empty = {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty((0, 4))}
    rpn_batches = []
    student.rpn.register_forward_pre_hook(lambda _, args: rpn_batches.append(len(args[0].image_sizes)))
    loss, extra = soft_teacher_unsupervised_loss(
        student, teacher, [img, img], [empty, target], [False, True],
        [torch.empty(0, dtype=torch.bool), pseudo[0]["reg_valid"]], cache
    )
    assert rpn_batches == [1]
    assert loss is not None and torch.isfinite(loss).item()
    loss.backward()
    assert student.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_core_ssod_v2_grad_accum_cross_microbatch_resume_and_plots(tmp_path):
    """Verify CoRe-SSOD + FasterRCNN-v2 + warmup_cosine + Kaggle(1x2) -> 5090(2x1) resume + conformal plots."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from make_synthetic_bdd import create
    from plot_results import render
    from prepare_bdd100k import main as prepare_main
    from ssod_research.native_training import parse_args, train
    from ssod_research.recovery import read_jsonl

    src = tmp_path / "src"
    paths = create(src, n_train=14, n_val=4, seed=42)
    split_dir = tmp_path / "split"
    prepare_main([
        "--train-labels", paths["train_labels"],
        "--val-labels", paths["val_labels"],
        "--output", str(split_dir),
        "--fraction", "0.5",
        "--seed", "42",
    ])

    run_dir = tmp_path / "runs" / "10pct_s42" / "core_ssod"

    def _argv(batch_size: int, grad_accum: int, stop_after: int = 0):
        return [
            "--data-root", str(split_dir),
            "--train-images", paths["train_images"],
            "--val-images", paths["val_images"],
            "--output", str(run_dir),
            "--variant", "core_ssod",
            "--detector", "fasterrcnn_v2",
            "--seed", "42",
            "--steps", "2",
            "--batch-size", str(batch_size),
            "--grad-accum-steps", str(grad_accum),
            "--max-side", "96",
            "--lr", "0.01",
            "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "1",
            "--conformal-every", "1",
            "--conformal-max-cal", "2",
            "--save-every", "1",
            "--eval-every", "2",
            "--eval-max-images", "2",
            "--device", "cpu",
            "--amp", "off",
            "--no-pretrained-backbone",
            "--stop-after-steps", str(stop_after),
        ]

    # Step 1 on "Kaggle" with micro-batch 1 x grad-accum 2 = effective batch 2
    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_argv(batch_size=1, grad_accum=2, stop_after=1)))

    # Resume step 2 on "RTX 5090" with micro-batch 2 x grad-accum 1 = SAME effective batch 2!
    meta = train(parse_args(_argv(batch_size=2, grad_accum=1, stop_after=0)))
    assert meta["status"] == "completed"
    assert [r["step"] for r in read_jsonl(run_dir / "steps.jsonl")] == [1, 2]

    # Verify conformal calibration diagnostics were logged and render cleanly
    conf_rows = read_jsonl(run_dir / "conformal_calibration.jsonl")
    assert len(conf_rows) >= 2
    assert "clear" in conf_rows[0]["regimes"] and "degraded" in conf_rows[0]["regimes"]
    fig_dir = render(tmp_path)
    assert list(fig_dir.glob("conformal_*.png"))


def test_q1_presets_and_strong_baselines_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from make_synthetic_bdd import create
    from prepare_bdd100k import main as prepare_main
    from run_suite import generate_jobs, parse_args as suite_parse
    from ssod_research.native_training import parse_args, train

    # Verify Q1 preset structure AND legacy core matrix preservation
    core_jobs = generate_jobs(suite_parse(["--preset", "core", "--output", str(tmp_path / "core")]))
    assert len(core_jobs) == 52

    q1_pilot_jobs = generate_jobs(suite_parse(["--preset", "q1_pilot", "--output", str(tmp_path / "q1p")]))
    q1p_names = [j["name"] for j in q1_pilot_jobs]
    for expected in ("10pct_s42/supervised", "10pct_s42/soft_teacher", "10pct_s42/consistent_teacher", "10pct_s42/core_ssod"):
        assert expected in q1p_names

    q1_paper_jobs = generate_jobs(suite_parse(["--preset", "q1_paper", "--output", str(tmp_path / "q1")]))
    q1_names = [j["name"] for j in q1_paper_jobs]
    assert "05pct_s42/core_no_vis_weight" in q1_names
    assert "05pct_s42/core_alpha005" in q1_names
    assert "05pct_s123/core_no_vis_weight" not in q1_names
    assert "10pct_s2026/soft_teacher" in q1_names
    assert "10pct_s2026/core_ssod" in q1_names

    # 1-step end-to-end smoke check for both soft_teacher and consistent_teacher
    src = tmp_path / "src"
    paths = create(src, n_train=10, n_val=2, seed=42)
    split_dir = tmp_path / "split"
    prepare_main([
        "--train-labels", paths["train_labels"],
        "--val-labels", paths["val_labels"],
        "--output", str(split_dir),
        "--fraction", "0.4",
        "--seed", "42",
    ])
    for variant in ("soft_teacher", "consistent_teacher"):
        meta = train(parse_args([
            "--data-root", str(split_dir),
            "--train-images", paths["train_images"],
            "--val-images", paths["val_images"],
            "--output", str(tmp_path / f"run_{variant}"),
            "--variant", variant,
            "--steps", "1",
            "--max-side", "96",
            "--eval-every", "1",
            "--eval-max-images", "2",
            "--device", "cpu",
            "--amp", "off",
            "--no-pretrained-backbone",
        ]))
        assert meta["status"] == "completed"
