"""Mandatory end-to-end crash-safety, resumability, and portability tests.

Covers all 12 recovery scenarios required by Section 10:
 1. Interruption during training
 2. Resume from a saved checkpoint + deterministic CPU weight/optimizer equivalence
 3. Corrupted latest checkpoint automatic fallback
 4. Missing checkpoint safe recovery
 5. Configuration mismatch detection
 6. Dataset fingerprint mismatch detection
 7. Interrupted evaluation resumption
 8. Interrupted preprocessing recovery
 9. Repeated execution of a completed pipeline (idempotency)
10. Dependency-aware selective downstream invalidation
11. Stale RUNNING status recovery after process crash / cloud timeout
12. Concurrent execution lock protection + multi-machine lease
13. Codex session restart continuity
14. Portable Kaggle -> RTX 5090 bundle export/import & cross-GPU compatibility
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ssod_research.cloud_sync import (
    MachineLease,
    export_portable_bundle,
    import_portable_bundle,
    validate_transfer_compatibility,
)
from ssod_research.pipeline import PipelineDAG, StageNode, inspect_codex_session
from ssod_research.recovery import FileLock, IncompatibleCheckpoint, LockBusy, read_json, read_jsonl
from ssod_research.registry import ExperimentRegistry, make_experiment_id


def _make_split(tmp_path: Path):
    """Create a tiny synthetic BDD dataset and converted COCO split."""
    from make_synthetic_bdd import create
    from prepare_bdd100k import main as prepare_main

    src = tmp_path / "synthetic_src"
    paths = create(src, n_train=10, n_val=4, seed=42)
    split_dir = tmp_path / "split_10pct_s42"
    prepare_main([
        "--train-labels", paths["train_labels"],
        "--val-labels", paths["val_labels"],
        "--output", str(split_dir),
        "--fraction", "0.4",
        "--seed", "42",
    ])
    return paths, split_dir


def _train_argv(split_dir: Path, paths: dict, out_dir: Path, **overrides) -> list[str]:
    cfg = {
        "data-root": str(split_dir),
        "train-images": paths["train_images"],
        "val-images": paths["val_images"],
        "output": str(out_dir),
        "variant": "mean_teacher",
        "seed": "42",
        "steps": "4",
        "batch-size": "1",
        "max-side": "96",
        "lr": "0.003",
        "save-every": "2",
        "eval-every": "4",
        "eval-max-images": "2",
        "device": "cpu",
        "amp": "off",
    }
    cfg.update({k.replace("_", "-"): str(v) for k, v in overrides.items()})
    argv = []
    for k, v in cfg.items():
        argv.extend([f"--{k}", v])
    argv.extend(["--no-pretrained-backbone", "--deterministic"])
    return argv


# ------------------------------------------------------------------ 1 & 2: deterministic interrupt & exact resume
def test_deterministic_cpu_interrupt_and_resume_equivalence(tmp_path):
    """Compare uninterrupted 4-step CPU run vs interrupted-at-2 and resumed run.

    Documented tolerance: atol=1e-5, rtol=1e-5 for student weights,
    teacher/EMA weights, and SGD optimizer momentum buffers on deterministic CPU.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_training import parse_args, train
    from ssod_research.recovery import safe_torch_load

    paths, split_dir = _make_split(tmp_path)
    out_full = tmp_path / "run_uninterrupted"
    out_res = tmp_path / "run_resumed"

    # Uninterrupted 4-step reference run
    train(parse_args(_train_argv(split_dir, paths, out_full, steps=4, save_every=2, eval_every=4)))

    # Interrupted run: stops cleanly after step 2
    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_train_argv(
            split_dir, paths, out_res, steps=4, save_every=2, eval_every=4, stop_after_steps=2,
        )))

    m_mid = read_json(out_res / "manifest.json")
    assert m_mid["status"] == "interrupted"
    assert m_mid["last_completed_step"] == 2
    assert (out_res / "checkpoints" / "ckpt_000000002.pth").is_file()
    assert (out_res / "checkpoints" / "ckpt_000000002.json").is_file()

    # Resume from step 2 -> step 4
    m_end = train(parse_args(_train_argv(split_dir, paths, out_res, steps=4, save_every=2, eval_every=4)))
    assert m_end["status"] == "completed"
    assert m_end["resume_mode"] == "exact"
    assert m_end["resume_warnings"] == []

    # Step logs must have exact steps 1..4 with zero duplicates
    step_nums = [r["step"] for r in read_jsonl(out_res / "steps.jsonl")]
    assert step_nums == [1, 2, 3, 4]

    # Compare final student, teacher/EMA, optimizer momentum buffers & LR scheduler
    st_full = safe_torch_load(out_full / "checkpoints" / "ckpt_000000004.pth")
    st_res = safe_torch_load(out_res / "checkpoints" / "ckpt_000000004.pth")

    assert st_full["step"] == st_res["step"] == 4
    assert st_full["scheduler"]["last_epoch"] == st_res["scheduler"]["last_epoch"] == 4

    for key in st_full["student"]:
        a, b = st_full["student"][key], st_res["student"][key]
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        else:
            assert torch.equal(a, b)

    for key in st_full["teacher"]:
        a, b = st_full["teacher"][key], st_res["teacher"][key]
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        else:
            assert torch.equal(a, b)

    opt_a = st_full["optimizer"]["state"]
    opt_b = st_res["optimizer"]["state"]
    assert set(opt_a.keys()) == set(opt_b.keys())
    for pid in opt_a:
        buf_a = opt_a[pid].get("momentum_buffer")
        buf_b = opt_b[pid].get("momentum_buffer")
        if buf_a is not None:
            torch.testing.assert_close(buf_a, buf_b, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ 3: corrupted latest checkpoint fallback
def test_corrupted_latest_checkpoint_falls_back_to_previous(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_training import parse_args, train

    paths, split_dir = _make_split(tmp_path)
    out = tmp_path / "run_corrupt"

    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_train_argv(
            split_dir, paths, out, variant="supervised", steps=3, save_every=1, eval_every=3, stop_after_steps=2,
        )))

    ckpt1 = out / "checkpoints" / "ckpt_000000001.pth"
    ckpt2 = out / "checkpoints" / "ckpt_000000002.pth"
    assert ckpt1.is_file() and ckpt2.is_file()

    # Corrupt latest checkpoint AND last.pth
    ckpt2.write_bytes(b"CORRUPTED_BYTES_SIMULATING_POWER_LOSS")
    (out / "last.pth").write_bytes(b"CORRUPTED_LAST")

    meta = train(parse_args(_train_argv(
        split_dir, paths, out, variant="supervised", steps=3, save_every=1, eval_every=3,
    )))
    assert meta["status"] == "completed"
    assert any("quarantined" in note for note in meta["recovery_notes"])
    assert (out / "checkpoints" / "quarantine.jsonl").is_file()
    assert list((out / "checkpoints").glob("ckpt_000000002.pth.corrupt-*"))
    assert [r["step"] for r in read_jsonl(out / "steps.jsonl")] == [1, 2, 3]


# ------------------------------------------------------------------ 4: missing checkpoint recovery
def test_missing_checkpoint_restarts_safely_without_claiming_exact_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_training import parse_args, train

    paths, split_dir = _make_split(tmp_path)
    out = tmp_path / "run_missing_ckpt"

    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_train_argv(
            split_dir, paths, out, variant="supervised", steps=2, save_every=1, eval_every=2, stop_after_steps=1,
        )))

    # Delete checkpoints and last.pth while keeping manifest.json + steps.jsonl
    shutil.rmtree(out / "checkpoints")
    (out / "last.pth").unlink()
    if (out / "last.json").exists():
        (out / "last.json").unlink()

    meta = train(parse_args(_train_argv(
        split_dir, paths, out, variant="supervised", steps=2, save_every=1, eval_every=2,
    )))
    assert meta["status"] == "completed"
    assert meta["resume_mode"] == "fresh_after_missing_checkpoint"
    assert [r["step"] for r in read_jsonl(out / "steps.jsonl")] == [1, 2]


# ------------------------------------------------------------------ 5 & 6: config & dataset mismatch
def test_configuration_and_dataset_mismatch_refuse_unsafe_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_training import parse_args, train

    paths, split_dir = _make_split(tmp_path)
    out = tmp_path / "run_mismatch"

    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_train_argv(
            split_dir, paths, out, variant="supervised", steps=2, save_every=1, eval_every=2, stop_after_steps=1, lr=0.003,
        )))

    # Config mismatch (changed learning rate)
    with pytest.raises(IncompatibleCheckpoint):
        train(parse_args(_train_argv(
            split_dir, paths, out, variant="supervised", steps=2, save_every=1, eval_every=2, lr=0.01,
        )))

    # Dataset fingerprint mismatch (mutate labeled.json)
    labeled_json = split_dir / "annotations" / "labeled.json"
    obj = json.loads(labeled_json.read_text(encoding="utf8"))
    obj["info"]["description"] = "mutated dataset"
    labeled_json.write_text(json.dumps(obj), encoding="utf8")

    with pytest.raises(IncompatibleCheckpoint):
        train(parse_args(_train_argv(
            split_dir, paths, out, variant="supervised", steps=2, save_every=1, eval_every=2, lr=0.003,
        )))


# ------------------------------------------------------------------ 7: interrupted evaluation
def test_interrupted_evaluation_resumes_from_partial_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_data import RoadCocoDataset
    from ssod_research.native_training import build_model, evaluate, seed_all

    paths, split_dir = _make_split(tmp_path)
    val_ds = RoadCocoDataset(split_dir / "annotations" / "val.json", paths["val_images"], max_side=96)
    seed_all(42)
    model = build_model(pretrained_backbone=False, max_side=96)
    device = torch.device("cpu")

    # Reference uninterrupted evaluation
    ref_metrics = evaluate(model, val_ds, device, tmp_path / "eval_ref", eval_fingerprint="fp1")

    # Interrupted evaluation after 2 of 4 images
    eval_res_dir = tmp_path / "eval_res"
    with pytest.raises(InterruptedError):
        evaluate(model, val_ds, device, eval_res_dir, eval_fingerprint="fp1", stop_after_images=2)

    assert (eval_res_dir / "partial_predictions.jsonl").is_file()
    assert len(read_jsonl(eval_res_dir / "partial_predictions.jsonl")) == 2
    assert not (eval_res_dir / "metrics.json").exists()

    # Resume evaluation -> completes remaining 2 images only
    res_metrics = evaluate(model, val_ds, device, eval_res_dir, eval_fingerprint="fp1", resume_partial=True)
    assert res_metrics["complete"] is True
    assert not (eval_res_dir / "partial_predictions.jsonl").exists()
    assert res_metrics["images"] == ref_metrics["images"] == 4
    assert res_metrics["AP"] == pytest.approx(ref_metrics["AP"])


# ------------------------------------------------------------------ 8 & 9: interrupted preprocessing + idempotent repeat
def test_interrupted_preprocessing_and_idempotent_pipeline_execution(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from run_suite import generate_jobs, job_complete, main as suite_main, parse_args

    out = tmp_path / "smoke_suite"
    argv = [
        "--preset", "smoke", "--device", "cpu", "--amp", "off",
        "--max-side", "96", "--warmup-steps", "1", "--train-steps", "1",
        "--output", str(out), "--no-update-workspace-docs",
    ]
    suite_main(argv)

    # Verify SQLite registry & progress artifacts
    reg = ExperimentRegistry(out / "experiment_state.sqlite")
    stages = reg.list_stages(preset="smoke")
    assert len(stages) > 0
    assert all(s["status"] == "COMPLETED" for s in stages)
    assert (out / "progress.json").is_file()
    assert (out / "runtime_estimates.json").is_file()
    assert (out / "PROJECT_PROGRESS.md").is_file()

    # Record mtimes of all checkpoints to prove second run does ZERO re-training
    ckpts = list((out / "runs").rglob("last.pth"))
    assert len(ckpts) == 4  # warmup + control + teacher + combined
    mtimes_before = {p: p.stat().st_mtime_ns for p in ckpts}

    # Repeat execution -> must skip every completed stage idempotently
    suite_main([*argv, "--resume"])
    mtimes_after = {p: p.stat().st_mtime_ns for p in ckpts}
    assert mtimes_before == mtimes_after

    # Simulate interrupted preprocessing: corrupt split_manifest.json and labeled.json
    split_dir = out / "splits" / "10pct_s42"
    (split_dir / "split_manifest.json").unlink()
    (split_dir / "annotations" / "labeled.json").write_text("{corrupt", encoding="utf8")

    args = parse_args([
        *argv,
        "--train-labels", str(out / "synthetic_source" / "synthetic_train_labels.json"),
        "--val-labels", str(out / "synthetic_source" / "synthetic_val_labels.json"),
        "--train-images", str(out / "synthetic_source" / "images" / "train"),
        "--val-images", str(out / "synthetic_source" / "images" / "val"),
    ])
    prep_job = generate_jobs(args)[0]
    assert job_complete(prep_job) is False

    # Re-run suite -> recovers preprocessing cleanly while reusing intact downstream runs
    suite_main([*argv, "--resume", "--run-limit", "1"])
    assert (split_dir / "split_manifest.json").is_file()
    assert job_complete(prep_job) is True


# ------------------------------------------------------------------ 10: dependency-aware selective invalidation
def test_dependency_graph_selective_invalidation(tmp_path):
    reg = ExperimentRegistry(tmp_path / "dag.sqlite")

    def _build_dag(s42_warmup_fp: str) -> PipelineDAG:
        nodes = [
            StageNode("p__s42_prep", "p", "s42/prepare", "dataset_splitting", 0, [], config_fingerprint="prep42"),
            StageNode("p__s42_warm", "p", "s42/warmup", "supervised_warmup", 1, ["p__s42_prep"], config_fingerprint=s42_warmup_fp),
            StageNode("p__s42_full", "p", "s42/full", "proposed_training", 2, ["p__s42_prep", "p__s42_warm"], config_fingerprint="full42"),
            StageNode("p__s123_prep", "p", "s123/prepare", "dataset_splitting", 3, [], config_fingerprint="prep123"),
            StageNode("p__s123_warm", "p", "s123/warmup", "supervised_warmup", 4, ["p__s123_prep"], config_fingerprint="warm123"),
            StageNode("p__s123_full", "p", "s123/full", "proposed_training", 5, ["p__s123_prep", "p__s123_warm"], config_fingerprint="full123"),
            StageNode("p__report", "p", "final_report", "final_report", 6, ["p__s42_full", "p__s123_full"], config_fingerprint="rep"),
        ]
        return PipelineDAG(nodes)

    dag1 = _build_dag("warm42_v1")
    dag1.reconcile_with_registry(reg)
    for node in dag1.topological_order():
        reg.mark_completed(node.experiment_id)

    # Change ONLY s42 warmup config -> must invalidate s42/warmup, s42/full, final_report ONLY
    dag2 = _build_dag("warm42_v2")
    res = dag2.reconcile_with_registry(reg)
    inv_ids = {item["experiment_id"] for item in res["invalidated"]}
    assert inv_ids == {"p__s42_warm", "p__s42_full", "p__report"}

    # Unrelated seed s123 and s42/prepare MUST remain COMPLETED
    for untouched in ("p__s42_prep", "p__s123_prep", "p__s123_warm", "p__s123_full"):
        assert reg.get_stage(untouched)["status"] == "COMPLETED"
    for invalidated in ("p__s42_warm", "p__s42_full", "p__report"):
        assert reg.get_stage(invalidated)["status"] == "PENDING"


# ------------------------------------------------------------------ 11: stale RUNNING recovery
def test_stale_running_status_recovered_after_crash(tmp_path):
    reg = ExperimentRegistry(tmp_path / "stale.sqlite")
    reg.upsert_stage(
        experiment_id="core__local_crash", suite_preset="core",
        stage_name="05pct_s42/full", stage_type="proposed_training",
        output_dir=str(tmp_path / "local_run"),
    )
    reg.upsert_stage(
        experiment_id="core__remote_crash", suite_preset="core",
        stage_name="10pct_s42/full", stage_type="proposed_training",
        output_dir=str(tmp_path / "remote_run"),
    )
    # Mark local crash with guaranteed-dead PID
    reg.mark_running("core__local_crash", pid=99999999)
    # Mark remote cloud session with expired heartbeat
    reg.mark_running("core__remote_crash", pid=1234, host="kaggle-terminated-node")
    with reg._tx() as conn:
        conn.execute(
            "UPDATE experiments SET heartbeat_utc = '2020-01-01T00:00:00+00:00' WHERE experiment_id = ?",
            ("core__remote_crash",),
        )

    recovered = reg.recover_stale_running(stale_after_sec=60.0)
    recovered_ids = {r["experiment_id"] for r in recovered}
    assert recovered_ids == {"core__local_crash", "core__remote_crash"}
    for eid in recovered_ids:
        row = reg.get_stage(eid)
        assert row["status"] == "INTERRUPTED"
        assert row["retry_history"]


# ------------------------------------------------------------------ 12: concurrent execution lock & cloud lease
def test_concurrent_execution_blocked_by_lock_and_lease(tmp_path):
    lock_path = tmp_path / ".experiment.lock"
    lock1 = FileLock(lock_path, purpose="worker-1")
    lock1.acquire()
    try:
        lock2 = FileLock(lock_path, purpose="worker-2")
        with pytest.raises(LockBusy):
            lock2.acquire(timeout=0.0)
    finally:
        lock1.release()

    # After release, another worker can acquire cleanly
    lock3 = FileLock(lock_path, purpose="worker-3")
    lock3.acquire()
    lock3.release()

    # Cross-machine lease blocks another host while active
    sync_dir = tmp_path / "shared_drive"
    lease_a = MachineLease(sync_dir, ttl_sec=600.0, preset="core")
    lease_a.me = "kaggle-node-1"
    lease_a.acquire(gpu_name="Tesla T4")
    try:
        lease_b = MachineLease(sync_dir, ttl_sec=600.0, preset="core")
        lease_b.me = "windows-rtx5090"
        with pytest.raises(RuntimeError, match="actively leased"):
            lease_b.acquire(gpu_name="RTX 5090")
        # Explicit force override succeeds when operator confirms remote host died
        lease_b.acquire(gpu_name="RTX 5090", force=True)
        lease_b.release()
    finally:
        lease_a.release()


# ------------------------------------------------------------------ 13: Codex session restart continuity
def test_codex_session_restart_reconstructs_full_state(tmp_path):
    db_path = tmp_path / "experiment_state.sqlite"
    reg = ExperimentRegistry(db_path)
    reg.upsert_stage(experiment_id="core__s1", suite_preset="core", stage_name="05pct_s42/prepare", stage_type="dataset_splitting", stage_order=0)
    reg.upsert_stage(experiment_id="core__s2", suite_preset="core", stage_name="05pct_s42/warmup", stage_type="supervised_warmup", stage_order=1, target_steps=1000)
    reg.upsert_stage(experiment_id="core__s3", suite_preset="core", stage_name="05pct_s42/full", stage_type="proposed_training", stage_order=2, target_steps=3000)
    reg.mark_completed("core__s1")
    reg.mark_running("core__s2", pid=99999999)  # crashed prior session
    reg.save_suite_state("core", tmp_path / "core_runs", "python scripts/run_suite.py --preset core --resume", {})

    state = inspect_codex_session(ROOT, db_path=db_path)
    assert state["docs"]["CODEX_HANDOFF.md"]["exists"] is True
    assert state["docs"]["PROJECT_PROGRESS.md"]["exists"] is True
    assert state["docs"]["DECISIONS.md"]["exists"] is True
    assert state["recovered_stale"] == ["core__s2"]
    assert state["completed"] == ["05pct_s42/prepare"]
    assert state["next_stage"] == "05pct_s42/warmup"
    assert "--resume" in state["recommended_command"]


# ------------------------------------------------------------------ 14: portable Kaggle -> RTX 5090 bundle transfer
def test_portable_bundle_transfer_cross_machine_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from ssod_research.native_training import parse_args, train

    paths, split_dir = _make_split(tmp_path)
    kaggle_out = tmp_path / "kaggle_machine" / "core_runs"
    run_src = kaggle_out / "runs" / "10pct_s42" / "supervised"

    # Train 1 of 2 steps on "machine 1"
    with pytest.raises(KeyboardInterrupt):
        train(parse_args(_train_argv(
            split_dir, paths, run_src, variant="supervised", steps=2, save_every=1, eval_every=2, stop_after_steps=1,
        )))

    # Export portable .tar.gz bundle
    bundle_tgz = tmp_path / "transfer" / "ssod_bundle.tar.gz"
    exp = export_portable_bundle(kaggle_out, bundle_tgz, create_archive=True)
    assert Path(exp["archive_path"]).is_file()

    # Validate compatibility check explicitly reports cross-GPU approximate notice
    sidecar = read_json(run_src / "checkpoints" / "ckpt_000000001.json")
    compat = validate_transfer_compatibility(
        sidecar,
        target_device_type="cuda",
        target_amp="bf16",
        target_gpu_name="NVIDIA GeForce RTX 5090",
    )
    assert compat["compatible"] is True
    assert compat["resume_mode"] == "approximate"
    assert compat["warnings"]

    # Import onto a brand new path (simulating RTX 5090 machine)
    rtx5090_out = tmp_path / "rtx5090_machine" / "core_runs"
    imp = import_portable_bundle(
        bundle_tgz,
        rtx5090_out,
        target_device_type="cpu",
        target_amp="off",
        target_gpu_name=None,
    )
    assert imp["restored_files"] > 0

    # Resume seamlessly from step 1 -> step 2 on the new path (no path-based fingerprint mismatch!)
    run_dst = rtx5090_out / "runs" / "10pct_s42" / "supervised"
    meta = train(parse_args(_train_argv(
        split_dir, paths, run_dst, variant="supervised", steps=2, save_every=1, eval_every=2,
    )))
    assert meta["status"] == "completed"
    assert [r["step"] for r in read_jsonl(run_dst / "steps.jsonl")] == [1, 2]


# ------------------------------------------------------------------ 15: CLI --status, --dry-run, resumable benchmark
def test_status_dry_run_and_resumable_benchmark_never_launch_training(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from benchmark_gpu import run_benchmark
    from run_suite import main as suite_main

    out = tmp_path / "status_only"
    # Neither --status nor --dry-run requires real BDD data or launches training
    suite_main(["--preset", "core", "--status", "--output", str(out), "--no-update-workspace-docs"])
    assert (out / "progress.json").is_file()
    assert (out / "PROJECT_PROGRESS.md").is_file()
    assert not (out / "runs").exists()

    suite_main(["--preset", "core", "--dry-run", "--output", str(out), "--no-update-workspace-docs"])
    assert (out / "experiment_plan.json").is_file()
    assert not (out / "runs").exists()

    # Mark a stage completed and verify --status and --dry-run never invalidate it
    reg = ExperimentRegistry(out / "experiment_state.sqlite")
    reg.mark_completed("core__environment_verification")
    suite_main(["--preset", "core", "--status", "--output", str(out), "--no-update-workspace-docs"])
    suite_main(["--preset", "core", "--dry-run", "--output", str(out), "--no-update-workspace-docs"])
    assert reg.get_stage("core__environment_verification")["status"] == "COMPLETED"

    # Resumable benchmark caches results in SQLite
    db_path = out / "experiment_state.sqlite"
    b1 = run_benchmark(device="cpu", amp="off", max_side=96, batch_size=1, steps=1, warmup=0, db_path=db_path, output_dir=out)
    assert b1["cached"] is False
    assert b1["sec_per_step"] > 0
    b2 = run_benchmark(device="cpu", amp="off", max_side=96, batch_size=1, steps=1, warmup=0, db_path=db_path, output_dir=out)
    assert b2["cached"] is True
    assert b2["sec_per_step"] == b1["sec_per_step"]
