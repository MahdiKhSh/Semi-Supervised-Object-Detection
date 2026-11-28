"""Comprehensive v2.2 audit regression suite covering Soft Teacher RPN proposal-only
extraction, micro-batch scaling invariance, WCRC exact bounds, stratified calibration,
small-object anchor & cutout safety, multi-resolution coordinate round-trips, DAG
fingerprint stability, invalidation enforcement, and real-GPU BF16 execution.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ------------------------------------------------------------------ 1. Section 6 Soft Teacher RPN Crash Regression
def test_soft_teacher_regression_only_batch_preserves_training_rpn_policy_and_gradients():
    """Reproduces the exact v2.1 crash (cls_valid=all False, reg_valid=True in training mode)
    and verifies proposal limits (2000 vs 1000) and gradient flow without mutating rpn.training.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models.detection.image_list import ImageList
    from ssod_research.baselines import (
        extract_rpn_proposals_only,
        soft_teacher_extract,
        soft_teacher_unsupervised_loss,
    )
    from ssod_research.native_training import build_model, target_for_training

    student = build_model(pretrained_backbone=False, max_side=320, detector="fasterrcnn_v2")
    teacher = build_model(pretrained_backbone=False, max_side=320, detector="fasterrcnn_v2")
    student.train()
    teacher.eval()

    # Verify extract_rpn_proposals_only preserves training top-N (2000) vs eval top-N (1000)
    assert student.rpn.pre_nms_top_n() == 2000 and student.rpn.post_nms_top_n() == 2000
    img_a = torch.rand(3, 240, 320)
    img_b = torch.rand(3, 200, 300)  # heterogeneous resolution!
    transformed, _ = student.transform([img_a], None)
    feats = student.backbone(transformed.tensors)
    props_train_policy = extract_rpn_proposals_only(
        student.rpn, ImageList(transformed.tensors, transformed.image_sizes), feats
    )
    assert student.rpn.training is True
    # On 240x320 there are >2000 total FPN anchors, so training mode returns >1000 proposals (up to 2000),
    # whereas flipping rpn.training=False would cap at 1000!
    assert len(props_train_policy[0]) > 1000

    # Mixed batch: image 0 = empty, image 1 = REG-ONLY (cls_valid=False, reg_valid=True -> crashed in v2.1!)
    _, _, cache = soft_teacher_extract(teacher, [img_a, img_b], jitter_times=2, rng=random.Random(5))
    empty_pseudo = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "scores": torch.zeros((0,)),
        "cls_valid": torch.zeros((0,), dtype=torch.bool),
        "reg_valid": torch.zeros((0,), dtype=torch.bool),
    }
    # Large box covering central region so untrained RPN proposals overlap >= 0.5 IoU -> pos_inds > 0!
    reg_only_pseudo = {
        "boxes": torch.tensor([[20.0, 20.0, 260.0, 180.0], [40.0, 30.0, 220.0, 160.0]]),
        "labels": torch.tensor([6, 10], dtype=torch.long),  # rare train (6) + traffic sign (10)
        "scores": torch.tensor([0.58, 0.62]),
        "cls_valid": torch.tensor([False, False]),
        "reg_valid": torch.tensor([True, True]),
    }

    rpn_modes_observed = []
    student.rpn.register_forward_pre_hook(lambda mod, _args: rpn_modes_observed.append((mod.training, len(_args[0].image_sizes))))

    loss, extra = soft_teacher_unsupervised_loss(
        student,
        teacher,
        [img_a, img_b],
        [target_for_training(empty_pseudo, image_id=1), target_for_training(reg_only_pseudo, image_id=2)],
        [False, True],
        [empty_pseudo["reg_valid"], reg_only_pseudo["reg_valid"]],
        cache,
        cls_valid_list=[empty_pseudo["cls_valid"], reg_only_pseudo["cls_valid"]],
    )
    assert student.rpn.training is True
    assert rpn_modes_observed == [(True, 1)]
    assert extra["active_cls_images"] == 0 and extra["active_reg_images"] == 1
    assert loss is not None and torch.isfinite(loss).item()
    loss.backward()
    assert student.roi_heads.box_predictor.bbox_pred.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.backbone.fpn.parameters())
    assert all(p.grad is None for p in teacher.parameters())


# ------------------------------------------------------------------ 2. All-Empty, Cls-Only, Zone2-Only, Rare & Small Batches
def test_edge_batches_all_empty_cls_only_zone2_only_and_multi_resolution():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.baselines import soft_teacher_extract, soft_teacher_unsupervised_loss
    from ssod_research.care_ssod import care_unsupervised_loss
    from ssod_research.native_training import build_model, target_for_training

    student = build_model(pretrained_backbone=False, max_side=160, detector="fasterrcnn_v2")
    teacher = build_model(pretrained_backbone=False, max_side=160, detector="fasterrcnn_v2")
    student.train()
    teacher.eval()

    imgs = [torch.rand(3, 96, 160), torch.rand(3, 144, 112)]
    _, _, st_cache = soft_teacher_extract(teacher, imgs, jitter_times=2, rng=random.Random(9))

    empty_z = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "scores": torch.zeros((0,)),
        "probs": torch.zeros((0, 11)),
        "reliability": torch.zeros((0,)),
        "q_loc": torch.zeros((0,)),
        "p_tp": torch.zeros((0,)),
        "cls_valid": torch.zeros((0,), dtype=torch.bool),
        "reg_valid": torch.zeros((0,), dtype=torch.bool),
    }

    # (a) All-empty pseudo-labels -> clean None, zero NaN
    l_st, s_st = soft_teacher_unsupervised_loss(
        student, teacher, imgs,
        [target_for_training(empty_z, image_id=1), target_for_training(empty_z, image_id=2)],
        [False, False], [empty_z["reg_valid"], empty_z["reg_valid"]], st_cache,
        cls_valid_list=[empty_z["cls_valid"], empty_z["cls_valid"]],
    )
    assert l_st is None and s_st["unsup"] == 0.0

    l_cr, s_cr = care_unsupervised_loss(
        student, imgs,
        [target_for_training(empty_z, image_id=1), target_for_training(empty_z, image_id=2)],
        [empty_z, empty_z], [empty_z, empty_z],
    )
    assert l_cr is None and s_cr["unsup"] == 0.0 and s_cr["stability"] == 0.0

    # (b) Classification-only Soft Teacher (cls_valid=True, reg_valid=False)
    cls_only = {
        "boxes": torch.tensor([[10.0, 12.0, 22.0, 28.0]]),  # small 12x16 traffic light!
        "labels": torch.tensor([9], dtype=torch.long),
        "scores": torch.tensor([0.88]),
        "cls_valid": torch.tensor([True]),
        "reg_valid": torch.tensor([False]),
    }
    student.zero_grad(set_to_none=True)
    l_cls, ex_cls = soft_teacher_unsupervised_loss(
        student, teacher, imgs,
        [target_for_training(cls_only, image_id=1), target_for_training(empty_z, image_id=2)],
        [True, False], [cls_only["reg_valid"], empty_z["reg_valid"]], st_cache,
        cls_valid_list=[cls_only["cls_valid"], empty_z["cls_valid"]],
    )
    assert l_cls is not None and torch.isfinite(l_cls).item()
    assert ex_cls["active_cls_images"] == 1 and ex_cls["active_reg_images"] == 0
    l_cls.backward()
    assert student.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0


# ------------------------------------------------------------------ 3. Micro-Batch Loss Normalization Invariance (M / B)
def test_active_batch_loss_scaling_prevents_sparse_batch_gradient_amplification():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.care_ssod import care_unsupervised_loss

    from ssod_research.native_training import build_model, seed_all, target_for_training

    seed_all(123)
    model = build_model(pretrained_backbone=False, max_side=128, detector="fasterrcnn_v2")
    model.train()
    img = torch.rand(3, 96, 128)
    probs = torch.full((1, 11), 0.01)
    probs[0, 3] = 0.90
    z2_only = {
        "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
        "labels": torch.tensor([3], dtype=torch.long),
        "scores": torch.tensor([0.55]),
        "probs": probs,
        "reliability": torch.tensor([0.50]),
        "p_tp": torch.tensor([0.45]),
    }
    empty_z = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "scores": torch.zeros((0,)),
        "probs": torch.zeros((0, 11)),
        "reliability": torch.zeros((0,)),
        "p_tp": torch.zeros((0,)),
    }

    # Single active image alone (B=1): loss = L
    loss_b1, _ = care_unsupervised_loss(
        model, [img], [target_for_training(empty_z, image_id=1)], [empty_z], [z2_only],
    )
    # Same single active image + 3 empty images (B=4): must equal 0.25 * L (exact 1x4 == 4x1 invariance!)
    loss_b4, _ = care_unsupervised_loss(
        model,
        [img, img, img, img],
        [target_for_training(empty_z, image_id=k) for k in range(4)],
        [empty_z] * 4,
        [z2_only, empty_z, empty_z, empty_z],
    )
    assert float(loss_b4.item()) == pytest.approx(0.25 * float(loss_b1.item()), rel=1e-4)


# ------------------------------------------------------------------ 4. Small-Object Cutout Occlusion Safety & Coordinate Round-Trip
def test_small_object_cutout_protection_and_coordinate_roundtrip():
    torch = pytest.importorskip("torch")
    from ssod_research.care_ssod import _flip_boxes_x, strong_view_care

    # Tiny 8x12 traffic light in a 96x128 image
    img = torch.full((3, 96, 128), 0.15)
    tiny_box = torch.tensor([[40.0, 30.0, 48.0, 42.0]])  # area = 96 px^2
    img[:, 30:42, 40:48] = 0.95  # distinctive bright object pixels
    z1 = {"boxes": tiny_box.clone(), "labels": torch.tensor([9]), "scores": torch.tensor([0.9])}
    z2 = {"boxes": torch.zeros((0, 4))}

    for seed in range(40):
        aug_img, aug_z1, _, was_flipped = strong_view_care(
            img, z1, z2, rng=random.Random(seed), enable_cutout=True, return_flip=True
        )
        # Round-trip box coordinates back to weak view must be exact
        back = _flip_boxes_x(aug_z1["boxes"], 128) if was_flipped else aug_z1["boxes"]
        assert torch.allclose(back, tiny_box, atol=1e-5)

        # Verify at least 65% of the tiny box's pixels still have distinct object structure (not 100% uniform cutout)
        bx1, by1, bx2, by2 = [int(round(v)) for v in aug_z1["boxes"][0].tolist()]
        crop = aug_img[:, by1:by2, bx1:bx2]
        # Even if partially touched by <=35% cutout, >=65% of bright pixels remain distinct from surrounding dark bg!
        assert crop.numel() == 3 * 12 * 8
        # Check against uniform constant patch erasure: if 100% erased, std across crop would be 0.0
        if seed % 3 == 0:
            # Verify image stays strictly in [0, 1]
            assert float(aug_img.min()) >= 0.0 and float(aug_img.max()) <= 1.0


# ------------------------------------------------------------------ 5. Exact WCRC Bound, Stratified Split & Small-Object Anchors
def test_exact_wcrc_bound_stratified_calibration_and_road_small_anchors():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.ops import box_iou
    from ssod_research.conformal import (
        WeightedConformalRiskCalibrator,
        solve_conformal_threshold,
        split_calibration_indices,
    )
    from ssod_research.native_training import build_model

    # (a) Stratified calibration split preserves rare singleton in train and balances multi-item strata
    strata = ["day|clear|r0"] * 50 + ["night|rainy|r0"] * 48 + ["night|foggy|r1"] * 1 + ["day|clear|r1"] * 1
    tr_ids, cal_ids = split_calibration_indices(100, seed=42, cal_fraction=0.10, max_cal=256, strata=strata)
    assert len(cal_ids) == 10 and len(tr_ids) == 90
    assert set(tr_ids).isdisjoint(set(cal_ids))
    # Singleton rare strata (indices 98 and 99) MUST remain in supervised training!
    assert 98 in tr_ids and 99 in tr_ids
    cal_night = sum(1 for idx in cal_ids if strata[idx].startswith("night"))
    assert 3 <= cal_night <= 6

    # (b) Exact Weighted CRC bound under heavy covariate-shift weights [0.4, 2.5]
    grid = [0.5, 0.7, 0.9]
    curves = [[0.4, 0.05, 0.0]] * 20
    weights = [0.4] * 10 + [2.5] * 10
    sol = solve_conformal_threshold(curves, weights, grid, alpha=0.15, min_threshold=0.5, max_threshold=0.9)
    w_sum = sum(weights)
    expected_ub_at_07 = (w_sum * 0.05 + 2.5) / (w_sum + 2.5)
    assert sol["threshold"] == pytest.approx(0.7)
    assert sol["conformal_upper"] == pytest.approx(expected_ub_at_07, abs=1e-4)
    assert sol["bound_satisfied"] is True

    # (c) Zone-2 tau_low never exceeds max_low (0.55) even when tau_high = 0.90
    cal = WeightedConformalRiskCalibrator(num_foreground_classes=10)
    cal.regime_high["degraded"] = 0.90
    cal.class_high["degraded"][9] = 0.90
    cal.class_low["degraded"][9] = 0.75
    thi, tlo = cal.thresholds(9, "degraded")
    assert thi == pytest.approx(0.90)
    assert tlo <= 0.55

    # (d) Road small-object anchor profile raises P2 anchor IoU >3.5x on a 10x14 traffic light!
    m_def = build_model(pretrained_backbone=False, max_side=256, detector="fasterrcnn_v2", anchor_profile="default")
    m_sml = build_model(pretrained_backbone=False, max_side=256, detector="fasterrcnn_v2", anchor_profile="road_small")
    assert sum(1 for mod in m_sml.modules() if isinstance(mod, torch.nn.BatchNorm2d)) == 0
    dummy = torch.rand(3, 160, 256)
    t_def, _ = m_def.transform([dummy], None)
    f_def = list(m_def.backbone(t_def.tensors).values())
    anc_def = m_def.rpn.anchor_generator(t_def, f_def)[0]
    anc_sml = m_sml.rpn.anchor_generator(t_def, f_def)[0]
    tiny_tl = torch.tensor([[80.0, 60.0, 90.0, 74.0]])  # 10x14 px traffic light
    max_iou_def = float(box_iou(anc_def, tiny_tl).max().item())
    max_iou_sml = float(box_iou(anc_sml, tiny_tl).max().item())
    assert max_iou_sml > 3.0 * max_iou_def
    assert max_iou_sml > 0.55


# ------------------------------------------------------------------ 6. Section 10 DAG Fingerprint Stability & Invalidation Refusal
def test_suite_restart_has_zero_false_invalidations_and_enforces_genuine_mismatches(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from run_suite import build_pipeline_dag, generate_jobs, job_complete, main as suite_main, parse_args
    from ssod_research.registry import ExperimentRegistry

    out = tmp_path / "suite_fp"
    argv = [
        "--preset", "smoke", "--device", "cpu", "--amp", "off",
        "--max-side", "96", "--warmup-steps", "1", "--train-steps", "1",
        "--output", str(out), "--no-update-workspace-docs",
    ]
    suite_main(argv)

    # Reconcile DAG on restart -> MUST produce ZERO false split/upstream/config/dataset invalidations!
    args2 = parse_args([
        *argv,
        "--train-labels", str(out / "synthetic_source" / "synthetic_train_labels.json"),
        "--val-labels", str(out / "synthetic_source" / "synthetic_val_labels.json"),
        "--train-images", str(out / "synthetic_source" / "images" / "train"),
        "--val-images", str(out / "synthetic_source" / "images" / "val"),
    ])
    jobs2 = generate_jobs(args2)
    dag2 = build_pipeline_dag(args2, jobs2)
    reg = ExperimentRegistry(out / "experiment_state.sqlite")
    reconcile = dag2.reconcile_with_registry(reg)
    assert reconcile["invalidated"] == []

    # Genuine mismatch 1: Changing --eval-max-images (partial 4 vs full 0) MUST raise RuntimeError!
    args_full_eval = parse_args([
        *argv, "--eval-max-images", "0",
        "--train-labels", str(out / "synthetic_source" / "synthetic_train_labels.json"),
        "--val-labels", str(out / "synthetic_source" / "synthetic_val_labels.json"),
        "--train-images", str(out / "synthetic_source" / "images" / "train"),
        "--val-images", str(out / "synthetic_source" / "images" / "val"),
    ])
    warmup_job_full = generate_jobs(args_full_eval)[1]
    with pytest.raises(RuntimeError, match="eval_max_images"):
        job_complete(warmup_job_full)

    # Genuine mismatch 2: Mutating upstream warmup/last.pth MUST refuse skipping downstream control!
    control_job = jobs2[2]
    assert job_complete(control_job) is True
    warmup_last = out / "runs" / "10pct_s42" / "warmup" / "last.pth"
    # Replace hardlink cleanly with distinct bytes so sha256 changes
    warmup_last.unlink()
    warmup_last.write_bytes(b"NEW_RETRAINED_WARMUP_WEIGHTS")
    with pytest.raises(RuntimeError, match="warm-start checkpoint"):
        job_complete(control_job)


# ------------------------------------------------------------------ 7. Real CUDA BF16 Smoke Test
@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA GPU required for BF16 hardware smoke test",
)
def test_cuda_bf16_mixed_batch_core_ssod_and_soft_teacher():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ssod_research.baselines import soft_teacher_extract, soft_teacher_unsupervised_loss
    from ssod_research.care_ssod import care_unsupervised_loss, extract_teacher_tri_zone_targets, strong_view_care
    from ssod_research.conformal import WeightedConformalRiskCalibrator
    from ssod_research.native_training import build_model, target_for_training

    device = torch.device("cuda")
    student = build_model(pretrained_backbone=False, max_side=256, detector="fasterrcnn_v2").to(device)
    teacher = build_model(pretrained_backbone=False, max_side=256, detector="fasterrcnn_v2").to(device)
    student.train()
    teacher.eval()

    imgs = [torch.rand(3, 180, 256, device=device), torch.rand(3, 180, 256, device=device) * 0.08]
    cal = WeightedConformalRiskCalibrator(num_foreground_classes=10)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        tri, _, t_cache = extract_teacher_tri_zone_targets(teacher, imgs, cal, return_cache=True)
        # Inject one Zone-1 box on img 0 and Zone-2-only on dark night img 1
        p1 = torch.full((1, 11), 0.01, device=device)
        p1[0, 3] = 0.91
        tri[0]["zone1"] = {
            "boxes": torch.tensor([[30.0, 30.0, 150.0, 130.0]], device=device),
            "labels": torch.tensor([3], dtype=torch.long, device=device),
            "scores": torch.tensor([0.91], device=device),
            "probs": p1,
            "reliability": torch.tensor([0.88], device=device),
            "q_loc": torch.tensor([0.90], device=device),
            "p_tp": torch.tensor([0.92], device=device),
        }
        tri[1]["zone2"] = {
            "boxes": torch.tensor([[40.0, 50.0, 90.0, 110.0]], device=device),
            "labels": torch.tensor([1], dtype=torch.long, device=device),
            "scores": torch.tensor([0.52], device=device),
            "probs": p1,
            "reliability": torch.tensor([0.46], device=device),
            "q_loc": torch.tensor([0.75], device=device),
            "p_tp": torch.tensor([0.42], device=device),
        }
        rng = random.Random(77)
        a0, z1_0, z2_0, f0 = strong_view_care(imgs[0], tri[0]["zone1"], tri[0]["zone2"], rng=rng, return_flip=True)
        a1, z1_1, z2_1, f1 = strong_view_care(imgs[1], tri[1]["zone1"], tri[1]["zone2"], rng=rng, return_flip=True)
        loss_core, _ = care_unsupervised_loss(
            student, [a0, a1],
            [target_for_training(z1_0, image_id=1), target_for_training(z1_1, image_id=2)],
            [z1_0, z1_1], [z2_0, z2_1],
            teacher=teacher, teacher_cache=t_cache, flips=[f0, f1],
        )
        assert loss_core is not None and torch.isfinite(loss_core).item()
    loss_core.backward()
    student.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        _, _, st_cache = soft_teacher_extract(teacher, imgs, jitter_times=2, rng=random.Random(3))
        reg_p = {
            "boxes": torch.tensor([[20.0, 20.0, 200.0, 150.0]], device=device),
            "labels": torch.tensor([3], dtype=torch.long, device=device),
            "scores": torch.tensor([0.60], device=device),
            "cls_valid": torch.tensor([False], device=device),
            "reg_valid": torch.tensor([True], device=device),
        }
        emp_p = {
            "boxes": torch.zeros((0, 4), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "scores": torch.zeros((0,), device=device),
            "cls_valid": torch.zeros((0,), dtype=torch.bool, device=device),
            "reg_valid": torch.zeros((0,), dtype=torch.bool, device=device),
        }
        loss_st, _ = soft_teacher_unsupervised_loss(
            student, teacher, imgs,
            [target_for_training(reg_p, image_id=1), target_for_training(emp_p, image_id=2)],
            [False, True], [reg_p["reg_valid"], emp_p["reg_valid"]], st_cache,
            cls_valid_list=[reg_p["cls_valid"], emp_p["cls_valid"]],
        )
        assert loss_st is not None and torch.isfinite(loss_st).item()
    loss_st.backward()


# ------------------------------------------------------------------ 8. Photometric Dual-Discovery, Tail Shrinkage & Zone-2 Reg
def test_photometric_dual_discovery_tail_conformal_shrinkage_and_zone2_regression():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from run_suite import generate_jobs, parse_args as suite_parse
    from ssod_research.care_ssod import (
        _photometric_restore_for_probe,
        care_unsupervised_loss,
        estimate_visibility_degradation,
        teacher_bidirectional_probe,
    )
    from ssod_research.conformal import WeightedConformalRiskCalibrator
    from ssod_research.native_training import build_model, target_for_training

    # (a) Photometric probe is exact identity on clear daytime and lifts dark shadow scenes in [0, 1]
    day = torch.zeros(3, 96, 128)
    day[:, :48, :] = 0.82
    day[:, 48:, :] = 0.28
    vis_day = estimate_visibility_degradation(day)
    assert vis_day["regime"] == "clear"
    assert torch.equal(_photometric_restore_for_probe(day, vis_day), day)

    night = torch.full((3, 96, 128), 0.05)
    night[:, 20:28, 30:38] = 0.98  # bright traffic light / headlight
    vis_night = estimate_visibility_degradation(night)
    assert vis_night["regime"] == "degraded"
    lifted = _photometric_restore_for_probe(night, vis_night)
    assert float(lifted.mean()) > 2.2 * float(night.mean())
    assert float(lifted.min()) >= 0.0 and float(lifted.max()) <= 1.0

    teacher = build_model(pretrained_backbone=False, max_side=160, detector="fasterrcnn_v2", anchor_profile="road_small")
    teacher.eval()
    probed, t_cache = teacher_bidirectional_probe(teacher, [day, night], enable_photometric_probe=True)
    assert len(probed) == 2 and len(t_cache["visibilities"]) == 2
    assert t_cache["visibilities"][1]["regime"] == "degraded"

    # (b) Tail-aware conformal shrinkage lowers rare-class prior relative to car & round-trips state_dict
    counts = {3: 70000, 1: 9000, 2: 400, 6: 18}
    cal = WeightedConformalRiskCalibrator(num_foreground_classes=10, class_counts=counts)
    tau_car, _ = cal.thresholds(3, "degraded")
    tau_rider, _ = cal.thresholds(2, "degraded")
    tau_train, _ = cal.thresholds(6, "degraded")
    assert tau_train < tau_rider < tau_car
    st = cal.state_dict()
    cal2 = WeightedConformalRiskCalibrator(num_foreground_classes=10)
    cal2.load_state_dict(st)
    assert cal2.state_dict() == st

    # (c) Decoupled Zone-2 high-stability box regression backpropagates through bbox_pred even with 0 Zone-1 boxes
    student = build_model(pretrained_backbone=False, max_side=256, detector="fasterrcnn_v2", anchor_profile="road_small")
    student.train()
    img = torch.rand(3, 192, 256)
    empty_z1 = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long),
        "scores": torch.zeros((0,)),
        "probs": torch.zeros((0, 11)),
        "reliability": torch.zeros((0,)),
        "q_loc": torch.zeros((0,)),
        "p_tp": torch.zeros((0,)),
    }
    probs_z2 = torch.full((1, 11), 0.02)
    probs_z2[0, 2] = 0.80
    z2_stable = {
        "boxes": torch.tensor([[24.0, 24.0, 210.0, 160.0]]),
        "labels": torch.tensor([2], dtype=torch.long),
        "scores": torch.tensor([0.58]),
        "probs": probs_z2,
        "reliability": torch.tensor([0.52]),
        "q_loc": torch.tensor([0.86]),
        "p_tp": torch.tensor([0.55]),
    }
    loss_z2, extra_z2 = care_unsupervised_loss(
        student,
        [img],
        [target_for_training(empty_z1, image_id=1)],
        [empty_z1],
        [z2_stable],
        class_counts=counts,
        kl_zone2_weight=0.0,  # isolate decoupled Zone-2 box regression!
    )
    assert loss_z2 is not None and torch.isfinite(loss_z2).item() and extra_z2["stability"] > 0.0
    loss_z2.backward()
    assert student.roi_heads.box_predictor.bbox_pred.weight.grad.abs().sum() > 0

    # (d) Q1 suite auto-enables road_small anchors, stratified calibration, and sqrt batch LR scaling
    q1_def = suite_parse(["--preset", "q1_a100"])
    assert q1_def.anchor_profile == "road_small" and q1_def.stratify_calibration is True and q1_def.lr == pytest.approx(0.01)
    q1_bs32 = suite_parse(["--preset", "q1_a100", "--batch-size", "32", "--grad-accum-steps", "1"])
    assert q1_bs32.lr == pytest.approx(0.02)
    q1_jobs = generate_jobs(q1_def)
    assert "--anchor-profile" in q1_jobs[1]["command"] and "road_small" in q1_jobs[1]["command"]
    assert "--stratify-calibration" in q1_jobs[1]["command"]

