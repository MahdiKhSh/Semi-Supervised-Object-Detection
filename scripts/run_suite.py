#!/usr/bin/env python3
"""Crash-safe, dependency-aware, resumable SSOD experiment orchestrator.

Works identically on Windows RTX 5090, Linux L40s/A100, Kaggle, Google Colab,
and CPU CI.  All pipeline stages are persisted in a transactional SQLite
registry, verified by cryptographic fingerprints, and safely resumable after
power loss, OOM, Ctrl+C, cloud session timeout, or cross-GPU migration.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssod_research.cloud_sync import (
    MachineLease,
    backup_sqlite,
    detect_runtime_environment,
    sync_directory,
    sync_results_to_github,
)
from ssod_research.pipeline import (
    PipelineDAG,
    StageNode,
    classify_training_stage,
    compute_runtime_estimates,
    format_duration,
    write_progress_reports,
)
from ssod_research.recovery import (
    CheckpointManager,
    FileLock,
    atomic_write_json,
    atomic_write_text,
    host_id,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
)
from ssod_research.registry import ExperimentRegistry, make_experiment_id

VARIANTS = ["supervised", "mean_teacher", "reliability", "adaptive", "condition", "full"]
PAPER_SEEDS = (42, 123, 2026)
PAPER_FRACTIONS = (0.05, 0.10)


def _default_output_for_preset(preset: str) -> Path:
    mapping = {
        "smoke": ROOT / "work_dirs" / "smoke",
        "pilot": ROOT / "work_dirs" / "pilot_runs",
        "core": ROOT / "work_dirs" / "core_runs",
        "care": ROOT / "work_dirs" / "care_runs",
        "q1_a100": ROOT / "work_dirs" / "q1_a100_runs",
        "q1_pilot": ROOT / "work_dirs" / "q1_pilot_runs",
        "q1_paper": ROOT / "work_dirs" / "q1_paper_runs",
        "paper": ROOT / "work_dirs" / "paper_runs",
    }
    return mapping.get(preset, ROOT / "work_dirs" / f"{preset}_runs")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crash-safe resumable SSOD research orchestrator")
    p.add_argument(
        "--preset",
        choices=("smoke", "pilot", "core", "care", "q1_pilot", "q1_a100", "q1_paper", "paper"),
        default="paper",
    )
    p.add_argument("--include-care", action="store_true",
                   help="Append proposed CaRe-SSOD / CoRe-SSOD variant to legacy presets")
    p.add_argument("--train-labels", type=Path, default=None)
    p.add_argument("--val-labels", type=Path, default=None)
    p.add_argument("--train-images", type=Path, default=None)
    p.add_argument("--val-images", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--detector", choices=("fasterrcnn_v1", "fasterrcnn_v2"), default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--lr-schedule",
        choices=("constant", "multistep", "cosine", "warmup_cosine", "warmup_multistep"),
        default=None,
    )
    p.add_argument("--lr-warmup-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-last", action="store_true",
                   help="Resume the previously selected suite recorded in work_dirs/last_suite.json")
    p.add_argument("--clean-restart", action="store_true",
                   help="Explicitly discard prior state in --output and restart from scratch")
    p.add_argument("--status", action="store_true",
                   help="Display progress and estimates without launching training")
    p.add_argument("--dry-run", action="store_true",
                   help="Print full DAG plan and status without launching training")
    p.add_argument("--benchmark", action="store_true",
                   help="Include resumable GPU/CPU benchmark stage before training")
    p.add_argument("--benchmark-only", action="store_true",
                   help="Run only the resumable GPU/CPU benchmark stage (no training)")
    p.add_argument("--db-path", type=Path, default=None,
                   help="Override SQLite registry path (default: <output>/experiment_state.sqlite)")
    p.add_argument("--sync-dir", type=Path, default=None,
                   help="Persistent storage directory (e.g. Google Drive / Kaggle export) to sync checkpoints")
    p.add_argument("--force-lease", action="store_true",
                   help="Override an existing remote host lease on --sync-dir")
    p.add_argument("--sync-github", action="store_true",
                   help="Commit and push lightweight results/progress/figures to GitHub after completed stages")
    p.add_argument("--stale-timeout", type=float, default=600.0,
                   help="Seconds before an unreachable remote RUNNING stage is recovered as INTERRUPTED")
    p.add_argument("--update-workspace-docs", action=argparse.BooleanOptionalAction, default=None,
                   help="Mirror progress to docs/PROJECT_PROGRESS.md and work_dirs/ (auto when inside work_dirs/)")
    p.add_argument("--run-limit", type=int, default=0,
                   help="debugging: only execute first N primary jobs, not a full paper result")
    p.add_argument("--dataset-from-plan", type=Path,
                   help="Reuse only the four BDD input paths from a previous suite experiment_plan.json.")
    args = p.parse_args(argv)
    if args.dataset_from_plan:
        try:
            previous = json.loads(args.dataset_from_plan.read_text(encoding="utf8"))
            prepare = next(job["command"] for job in previous["jobs"] if job["type"] == "prepare")
            for key in ("train_labels", "val_labels", "train_images", "val_images"):
                flag = "--" + key.replace("_", "-")
                value = Path(prepare[prepare.index(flag) + 1])
                if getattr(args, key) is None:
                    setattr(args, key, value)
        except (OSError, ValueError, StopIteration, IndexError, TypeError, KeyError) as exc:
            p.error(f"Cannot read BDD input paths from {args.dataset_from_plan}: {exc}")

    if args.resume_last:
        saved = _load_last_suite_descriptor()
        if saved:
            args.preset = saved.get("preset", args.preset)
            if args.output is None and saved.get("output"):
                args.output = Path(saved["output"])
            saved_args = saved.get("args") or {}
            for key in ("train_labels", "val_labels", "train_images", "val_images"):
                if saved_args.get(key):
                    setattr(args, key, Path(saved_args[key]))
            for key in (
                "device", "amp", "batch_size", "grad_accum_steps", "detector",
                "lr", "lr_schedule", "lr_warmup_steps",
                "warmup_steps", "train_steps", "max_side", "save_every",
            ):
                if saved_args.get(key) is not None and getattr(args, key) in (None, "auto"):
                    setattr(args, key, saved_args[key])
            if args.sync_dir is None and saved_args.get("sync_dir"):
                args.sync_dir = Path(saved_args["sync_dir"])
        args.resume = True

    data_root = ROOT / "data/bdd100k"
    args.train_labels = args.train_labels or data_root / "raw/labels_train.json"
    args.val_labels = args.val_labels or data_root / "raw/labels_val.json"
    args.train_images = args.train_images or data_root / "images/train"
    args.val_images = args.val_images or data_root / "images/val"

    # Apply preset-aware defaults (safe on Kaggle T4 16GB and portable to RTX 5090 / L40s)
    is_q1 = args.preset in ("q1_pilot", "q1_a100", "q1_paper")
    if args.batch_size is None:
        args.batch_size = 2 if is_q1 else 1
    if args.grad_accum_steps is None:
        args.grad_accum_steps = 4 if is_q1 else 1  # effective batch = 8 for q1 presets!
    if args.detector is None:
        args.detector = "fasterrcnn_v2" if is_q1 else "fasterrcnn_v1"
    if args.lr is None:
        args.lr = 0.01 if is_q1 else 0.003
    if args.lr_schedule is None:
        args.lr_schedule = "warmup_cosine" if is_q1 else "constant"
    if args.lr_warmup_steps is None:
        args.lr_warmup_steps = 500 if args.preset == "q1_paper" else (150 if is_q1 else 0)

    if args.output is None:
        args.output = _default_output_for_preset(args.preset)

    if args.update_workspace_docs is None:
        try:
            args.output.resolve().relative_to((ROOT / "work_dirs").resolve())
            args.update_workspace_docs = True
        except ValueError:
            args.update_workspace_docs = False

    if args.db_path is None:
        args.db_path = args.output / "experiment_state.sqlite"

    return args


def _load_last_suite_descriptor() -> dict | None:
    last_json = ROOT / "work_dirs" / "last_suite.json"
    info = read_json(last_json)
    if info:
        return info
    ws_db = ROOT / "work_dirs" / "experiment_state.sqlite"
    if ws_db.is_file():
        return ExperimentRegistry(ws_db).load_suite_state()
    return None


def experiment_plan(preset, train_steps=None, warmup_steps=None, max_side=None):
    if preset == "smoke":
        specs = ((0.10, 42),)
        selected = [
            ("supervised", 0.7, "control", {}),
            ("mean_teacher", 0.7, "teacher", {}),
            ("full", 0.7, "combined", {}),
        ]
        return specs, selected, warmup_steps or 1, train_steps or 2, max_side or 256
    if preset == "pilot":
        specs = ((0.10, 42),)
        selected = [(v, 0.7, v, {}) for v in VARIANTS]
        selected += [
            ("full", 0.5, "threshold05", {}),
            ("full", 0.9, "threshold09", {}),
            ("full", 0.7, "no_box", {"box_weight": 0.0}),
            ("full", 0.7, "no_class", {"class_weight": 0.0}),
            ("full", 0.7, "no_conf", {"confidence_weight": 0.0}),
        ]
        return specs, selected, warmup_steps or 30, train_steps or 100, max_side or 384
    if preset == "core":
        specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
        selected = [(v, 0.7, v, {}) for v in ("supervised", "mean_teacher", "reliability", "full")]
        selected += [
            ("full", 0.5, "threshold05", {}),
            ("full", 0.9, "threshold09", {}),
            ("full", 0.7, "no_box", {"box_weight": 0.0}),
            ("full", 0.7, "no_class", {"class_weight": 0.0}),
            ("full", 0.7, "no_conf", {"confidence_weight": 0.0}),
            ("full", 0.7, "no_localization", {"localization_enabled": 0}),
            ("full", 0.7, "no_condition", {"condition_offset": 0.0}),
            ("full", 0.7, "no_extra_consistency", {"consistency_weight": 0.0}),
        ]
        return specs, selected, warmup_steps or 1000, train_steps or 3000, max_side or 512
    if preset == "care":
        specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
        selected = [
            ("supervised", 0.7, "supervised", {}),
            ("mean_teacher", 0.7, "mean_teacher", {}),
            ("full", 0.7, "full", {}),
            ("care_ssod", 0.7, "care_ssod", {}),
            ("care_ssod", 0.7, "care_no_trizone", {"care_tri_zone": 0}),
            ("care_ssod", 0.7, "care_no_nwd", {"care_nwd": 0}),
            ("care_ssod", 0.7, "care_no_vis_cal", {"care_visibility_cal": 0}),
            ("care_ssod", 0.7, "care_no_cutout", {"care_cutout": 0}),
        ]
        return specs, selected, warmup_steps or 1000, train_steps or 3000, max_side or 512
    if preset in ("q1_pilot", "q1_a100"):
        specs = ((0.10, 42),)
        selected = [
            ("supervised", 0.7, "supervised", {}),
            ("mean_teacher", 0.7, "mean_teacher", {}),
            ("soft_teacher", 0.7, "soft_teacher", {}),
            ("consistent_teacher", 0.7, "consistent_teacher", {}),
            ("core_ssod", 0.7, "core_ssod", {}),
        ]
        if preset == "q1_a100":
            return specs, selected, warmup_steps or 1500, train_steps or 3000, max_side or 800
        return specs, selected, warmup_steps or 500, train_steps or 1500, max_side or 800
    if preset == "q1_paper":
        specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
        selected = [
            ("supervised", 0.7, "supervised", {}),
            ("mean_teacher", 0.7, "mean_teacher", {}),
            ("soft_teacher", 0.7, "soft_teacher", {}),
            ("consistent_teacher", 0.7, "consistent_teacher", {}),
            ("core_ssod", 0.7, "core_ssod", {}),
            ("care_ssod", 0.7, "care_ssod", {}),
            ("core_ssod", 0.7, "core_no_vis_weight", {"conformal_vis_weight": 0}),
            ("core_ssod", 0.7, "core_no_trizone", {"care_tri_zone": 0}),
            ("core_ssod", 0.7, "core_no_nwd", {"care_nwd": 0}),
            ("core_ssod", 0.7, "core_alpha005", {"conformal_alpha_strict": 0.05}),
            ("core_ssod", 0.7, "core_alpha025", {"conformal_alpha_strict": 0.25}),
        ]
        return specs, selected, warmup_steps or 3000, train_steps or 12000, max_side or 800
    specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
    selected = [(v, 0.7, v, {}) for v in VARIANTS]
    selected += [
        ("full", 0.5, "threshold05", {}),
        ("full", 0.9, "threshold09", {}),
        ("full", 0.7, "no_box", {"box_weight": 0.0}),
        ("full", 0.7, "no_class", {"class_weight": 0.0}),
        ("full", 0.7, "no_conf", {"confidence_weight": 0.0}),
        ("full", 0.7, "no_localization", {"localization_scale": 0.0}),
        ("full", 0.7, "no_adaptive", {"adaptive_strength": 0.0}),
        ("full", 0.7, "no_condition", {"condition_offset": 0.0}),
        ("full", 0.7, "no_extra_consistency", {"consistency_weight": 0.0}),
    ]
    return specs, selected, warmup_steps or 2000, train_steps or 8000, max_side or 640


def _default_save_every(preset: str, override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    if preset in ("paper", "q1_paper"):
        return 250
    if preset in ("core", "care", "q1_pilot", "q1_a100"):
        return 200
    return 10


def generate_jobs(args):
    """Return the canonical ordered primary jobs (prepare + train) for a preset."""
    specs, variants, warmup_steps, train_steps, side = experiment_plan(
        args.preset, args.train_steps, args.warmup_steps, args.max_side)
    if getattr(args, "include_care", False) and not any(v[0] == "core_ssod" for v in variants):
        variants = list(variants) + [("core_ssod", 0.7, "core_ssod", {})]
    if not 0 < warmup_steps < 10_000_000 or not 0 < train_steps < 10_000_000:
        raise ValueError("step budgets must be positive")
    save_every = _default_save_every(args.preset, getattr(args, "save_every", None))
    amp_arg = getattr(args, "amp", "auto")
    if args.device == "cpu":
        amp_arg = "off"
    eval_every = (
        str(train_steps)
        if args.preset in ("smoke", "pilot", "q1_pilot")
        else ("3000" if args.preset == "q1_paper" else "2000")
    )
    if args.preset == "q1_a100":
        eval_every = "1500"
    reserve_calibration = any(variant == "core_ssod" for variant, *_ in variants)
    jobs = []
    for fraction, seed in specs:
        frac_tag = f"{round(100 * fraction):02d}pct"
        split = args.output / "splits" / f"{frac_tag}_s{seed}"
        root = args.output / "runs" / f"{frac_tag}_s{seed}"
        base = [
            "--data-root", str(split),
            "--train-images", str(args.train_images),
            "--val-images", str(args.val_images),
            "--device", args.device,
            "--detector", str(getattr(args, "detector", "fasterrcnn_v1")),
            "--batch-size", str(args.batch_size),
            "--grad-accum-steps", str(getattr(args, "grad_accum_steps", 1)),
            "--max-side", str(side),
            "--lr", str(getattr(args, "lr", 0.003)),
            "--lr-schedule", str(getattr(args, "lr_schedule", "constant")),
            "--lr-warmup-steps", str(getattr(args, "lr_warmup_steps", 0)),
            "--seed", str(seed),
            "--save-every", str(save_every),
            "--eval-every", eval_every,
            "--amp", amp_arg,
        ]
        if reserve_calibration:
            base.append("--reserve-calibration")
        if args.preset == "smoke":
            base += ["--no-pretrained-backbone", "--eval-max-images", "4"]
        elif args.preset == "pilot":
            base += ["--eval-max-images", "50"]
        elif args.preset == "q1_pilot":
            base += ["--eval-max-images", "500"]
        prepare_cmd = [
            sys.executable, str(ROOT / "scripts" / "prepare_bdd100k.py"),
            "--train-labels", str(args.train_labels),
            "--val-labels", str(args.val_labels),
            "--output", str(split),
            "--fraction", str(fraction),
            "--seed", str(seed),
        ]
        if args.train_images:
            prepare_cmd += ["--train-images", str(args.train_images)]
        if args.val_images:
            prepare_cmd += ["--val-images", str(args.val_images)]
        jobs.append({
            "name": f"{frac_tag}_s{seed}/prepare",
            "type": "prepare",
            "fraction": fraction,
            "seed": seed,
            "split_dir": str(split),
            "output": str(split / "split_manifest.json"),
            "command": prepare_cmd,
        })
        warmup_output = root / "warmup"
        warmup_cmd = [
            sys.executable, "-m", "ssod_research.native_training",
            "--variant", "supervised",
            "--steps", str(warmup_steps),
            "--output", str(warmup_output),
            *base,
        ]
        jobs.append({
            "name": f"{frac_tag}_s{seed}/warmup",
            "type": "train",
            "tag": "warmup",
            "variant": "supervised",
            "fraction": fraction,
            "seed": seed,
            "threshold": 0.7,
            "max_side": side,
            "kwargs": {},
            "split_dir": str(split),
            "output": str(warmup_output),
            "steps": warmup_steps,
            "command": warmup_cmd,
        })
        init = warmup_output / "last.pth"
        for variant, threshold, tag, kw in variants:
            if args.preset == "core" and seed != 42 and tag not in (
                "supervised", "mean_teacher", "reliability", "full", "care_ssod"
            ):
                continue
            if args.preset == "care" and seed != 42 and tag not in (
                "supervised", "mean_teacher", "full", "care_ssod"
            ):
                continue
            if args.preset == "q1_paper" and seed != 42 and tag not in (
                "supervised", "mean_teacher", "soft_teacher", "consistent_teacher", "core_ssod"
            ):
                continue
            output = root / tag
            run_cmd = [
                sys.executable, "-m", "ssod_research.native_training",
                "--variant", variant,
                "--steps", str(train_steps),
                "--output", str(output),
                "--init-checkpoint", str(init),
                "--pseudo-threshold", str(threshold),
                *base,
            ]
            for key, value in kw.items():
                run_cmd += ["--" + key.replace("_", "-"), str(value)]
            jobs.append({
                "name": f"{frac_tag}_s{seed}/{tag}",
                "type": "train",
                "tag": tag,
                "variant": variant,
                "fraction": fraction,
                "seed": seed,
                "threshold": threshold,
                "max_side": side,
                "kwargs": dict(kw),
                "split_dir": str(split),
                "output": str(output),
                "steps": train_steps,
                "command": run_cmd,
            })
    return jobs


def _source_dataset_fingerprint(args) -> str:
    hashes = {}
    for key, path in (("train_labels", args.train_labels), ("val_labels", args.val_labels)):
        if path and Path(path).is_file():
            hashes[key] = sha256_file(path)
        else:
            hashes[key] = "unverified"
    return stable_hash(hashes)


def _split_fingerprint(split_dir: Path, fallback_seed: int, fallback_fraction: float, src_fp: str) -> str:
    manifest = read_json(split_dir / "split_manifest.json")
    if manifest and manifest.get("split_fingerprint"):
        return str(manifest["split_fingerprint"])
    return stable_hash({"seed": fallback_seed, "fraction": fallback_fraction, "source": src_fp})


def build_pipeline_dag(args, jobs: list[dict] | None = None) -> PipelineDAG:
    """Build the full 15-stage dependency graph wrapping the preset's jobs."""
    if jobs is None:
        jobs = generate_jobs(args)
    _, _, _, _, side = experiment_plan(args.preset, args.train_steps, args.warmup_steps, args.max_side)
    src_fp = _source_dataset_fingerprint(args)
    nodes: list[StageNode] = []
    order = 0

    env_id = make_experiment_id(args.preset, "environment_verification")
    nodes.append(StageNode(
        experiment_id=env_id,
        suite_preset=args.preset,
        stage_name="environment_verification",
        stage_type="environment_verification",
        order=order,
        dependencies=[],
        config={"device": args.device, "amp": args.amp},
        config_fingerprint=stable_hash({"stage": "environment_verification", "device": args.device}),
        output_dir=str(args.output),
    ))
    order += 1

    data_val_id = make_experiment_id(args.preset, "dataset_validation")
    nodes.append(StageNode(
        experiment_id=data_val_id,
        suite_preset=args.preset,
        stage_name="dataset_validation",
        stage_type="dataset_validation",
        order=order,
        dependencies=[env_id],
        config={"preset": args.preset},
        config_fingerprint=stable_hash({"stage": "dataset_validation", "preset": args.preset}),
        dataset_fingerprint=src_fp,
        output_dir=str(args.output),
    ))
    order += 1

    if args.benchmark or args.benchmark_only:
        bench_id = make_experiment_id(args.preset, "gpu_benchmarking")
        nodes.append(StageNode(
            experiment_id=bench_id,
            suite_preset=args.preset,
            stage_name="gpu_benchmarking",
            stage_type="gpu_benchmarking",
            order=order,
            dependencies=[env_id],
            config={"max_side": side, "batch_size": args.batch_size, "device": args.device, "amp": args.amp},
            config_fingerprint=stable_hash({"max_side": side, "batch_size": args.batch_size, "amp": args.amp}),
            output_dir=str(args.output),
        ))
        order += 1

    prepare_ids: dict[str, str] = {}
    warmup_ids: dict[str, str] = {}
    eval_ids: list[str] = []
    analysis_ids: list[str] = []

    for job in jobs:
        split_key = job["name"].split("/")[0]
        split_dir = Path(job["split_dir"])
        split_fp = _split_fingerprint(split_dir, int(job["seed"]), float(job["fraction"]), src_fp)

        if job["type"] == "prepare":
            prep_id = make_experiment_id(args.preset, job["name"])
            prepare_ids[split_key] = prep_id
            nodes.append(StageNode(
                experiment_id=prep_id,
                suite_preset=args.preset,
                stage_name=job["name"],
                stage_type="dataset_splitting",
                order=order,
                dependencies=[data_val_id],
                config={"fraction": job["fraction"], "seed": job["seed"]},
                config_fingerprint=stable_hash({"fraction": job["fraction"], "seed": job["seed"]}),
                dataset_fingerprint=src_fp,
                split_fingerprint=split_fp,
                seed=int(job["seed"]),
                label_fraction=float(job["fraction"]),
                output_dir=str(split_dir),
                payload={"job": job},
            ))
            order += 1
            continue

        # Training job
        tag = job["tag"]
        stage_type = classify_training_stage(job["variant"], tag)
        train_id = make_experiment_id(args.preset, job["name"])
        deps = [prepare_ids[split_key]]
        if tag == "warmup":
            warmup_ids[split_key] = train_id
        else:
            deps.append(warmup_ids[split_key])

        sci_cfg = {
            "variant": job["variant"],
            "tag": tag,
            "detector": str(getattr(args, "detector", "fasterrcnn_v1")),
            "seed": int(job["seed"]),
            "fraction": float(job["fraction"]),
            "threshold": float(job["threshold"]),
            "effective_batch_size": int(args.batch_size) * int(getattr(args, "grad_accum_steps", 1)),
            "max_side": int(job["max_side"]),
            "lr": float(getattr(args, "lr", 0.003)),
            "lr_schedule": str(getattr(args, "lr_schedule", "constant")),
            "lr_warmup_steps": int(getattr(args, "lr_warmup_steps", 0)),
            "steps": int(job["steps"]),
            "reserve_calibration": "--reserve-calibration" in job["command"],
            "pretrained_backbone": args.preset != "smoke",
            "kwargs": job.get("kwargs", {}),
        }
        cfg_fp = stable_hash(sci_cfg)
        nodes.append(StageNode(
            experiment_id=train_id,
            suite_preset=args.preset,
            stage_name=job["name"],
            stage_type=stage_type,
            order=order,
            dependencies=deps,
            config=sci_cfg,
            config_fingerprint=cfg_fp,
            dataset_fingerprint=src_fp,
            split_fingerprint=split_fp,
            model_variant=job["variant"],
            seed=int(job["seed"]),
            label_fraction=float(job["fraction"]),
            threshold=float(job["threshold"]),
            target_steps=int(job["steps"]),
            output_dir=str(job["output"]),
            payload={"job": job},
        ))
        order += 1

        # Explicit resumable evaluation stage
        eval_stage_name = f"{job['name']}/eval"
        eval_id = make_experiment_id(args.preset, eval_stage_name)
        eval_ids.append(eval_id)
        nodes.append(StageNode(
            experiment_id=eval_id,
            suite_preset=args.preset,
            stage_name=eval_stage_name,
            stage_type="model_evaluation",
            order=order,
            dependencies=[train_id],
            config={"steps": int(job["steps"]), "max_side": int(job["max_side"]), "preset": args.preset},
            config_fingerprint=stable_hash({"eval_steps": int(job["steps"]), "max_side": int(job["max_side"]), "preset": args.preset}),
            dataset_fingerprint=src_fp,
            split_fingerprint=split_fp,
            model_variant=job["variant"],
            seed=int(job["seed"]),
            label_fraction=float(job["fraction"]),
            threshold=float(job["threshold"]),
            output_dir=str(job["output"]),
            payload={"job": job},
        ))
        order += 1

        # Per-class and condition-stratified evaluation stage for core & paper
        if args.preset in ("core", "care", "q1_a100", "q1_paper", "paper"):
            an_name = f"{job['name']}/analysis"
            an_id = make_experiment_id(args.preset, an_name)
            analysis_ids.append(an_id)
            nodes.append(StageNode(
                experiment_id=an_id,
                suite_preset=args.preset,
                stage_name=an_name,
                stage_type="condition_per_class_evaluation",
                order=order,
                dependencies=[eval_id],
                config={"steps": int(job["steps"])},
                config_fingerprint=stable_hash({"analysis_steps": int(job["steps"])}),
                dataset_fingerprint=src_fp,
                split_fingerprint=split_fp,
                model_variant=job["variant"],
                seed=int(job["seed"]),
                label_fraction=float(job["fraction"]),
                threshold=float(job["threshold"]),
                output_dir=str(job["output"]),
                payload={"job": job},
            ))
            order += 1

    stat_id = make_experiment_id(args.preset, "statistical_analysis")
    nodes.append(StageNode(
        experiment_id=stat_id,
        suite_preset=args.preset,
        stage_name="statistical_analysis",
        stage_type="statistical_analysis",
        order=order,
        dependencies=eval_ids + analysis_ids,
        config={"preset": args.preset},
        config_fingerprint=stable_hash({"stage": "statistical_analysis", "preset": args.preset}),
        dataset_fingerprint=src_fp,
        output_dir=str(args.output),
    ))
    order += 1

    plot_id = make_experiment_id(args.preset, "plot_generation")
    nodes.append(StageNode(
        experiment_id=plot_id,
        suite_preset=args.preset,
        stage_name="plot_generation",
        stage_type="plot_generation",
        order=order,
        dependencies=[stat_id],
        config={"preset": args.preset},
        config_fingerprint=stable_hash({"stage": "plot_generation", "preset": args.preset}),
        dataset_fingerprint=src_fp,
        output_dir=str(args.output),
    ))
    order += 1

    rep_id = make_experiment_id(args.preset, "final_report")
    nodes.append(StageNode(
        experiment_id=rep_id,
        suite_preset=args.preset,
        stage_name="final_report",
        stage_type="final_report",
        order=order,
        dependencies=[stat_id, plot_id],
        config={"preset": args.preset},
        config_fingerprint=stable_hash({"stage": "final_report", "preset": args.preset}),
        dataset_fingerprint=src_fp,
        output_dir=str(args.output),
    ))

    return PipelineDAG(nodes)


def prepare_smoke_inputs(args):
    from make_synthetic_bdd import create
    source = args.output / "synthetic_source"
    train_labels = source / "synthetic_train_labels.json"
    val_labels = source / "synthetic_val_labels.json"
    train_images = source / "images" / "train"
    val_images = source / "images" / "val"
    if not (train_labels.is_file() and val_labels.is_file() and train_images.is_dir() and val_images.is_dir()):
        paths = create(source, seed=42)
        args.train_labels = Path(paths["train_labels"])
        args.val_labels = Path(paths["val_labels"])
        args.train_images = Path(paths["train_images"])
        args.val_images = Path(paths["val_images"])
    else:
        args.train_labels = train_labels
        args.val_labels = val_labels
        args.train_images = train_images
        args.val_images = val_images


def validate_input_files(args):
    for path in (args.train_labels, args.val_labels):
        if not path.is_file():
            raise FileNotFoundError(f"Missing BDD labels: {path}. See docs/WINDOWS_ONE_CLICK.md.")
    for directory in (args.train_images, args.val_images):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing BDD image folder: {directory}. See docs/WINDOWS_ONE_CLICK.md.")


def _flag(command, name):
    name = "--" + name.replace("_", "-")
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def job_complete(job):
    """Verify real, non-corrupted completion on disk (never trusts folder existence alone)."""
    if job["type"] == "prepare":
        from prepare_bdd100k import verify_split_output
        split_dir = Path(job["output"]).parent
        train_labels = Path(_flag(job["command"], "train_labels"))
        val_labels = Path(_flag(job["command"], "val_labels"))
        train_images_flag = _flag(job["command"], "train_images")
        val_images_flag = _flag(job["command"], "val_images")
        fraction = float(_flag(job["command"], "fraction"))
        seed = int(_flag(job["command"], "seed"))
        ok, reason, manifest = verify_split_output(
            split_dir,
            train_labels=train_labels,
            val_labels=val_labels,
            fraction=fraction,
            seed=seed,
            train_images=Path(train_images_flag) if train_images_flag else None,
            val_images=Path(val_images_flag) if val_images_flag else None,
        )
        if not ok and manifest is not None and ("mismatch" in reason or "source labels changed" in reason):
            raise RuntimeError(f"Split mismatch for {split_dir}: {reason}")
        return ok

    out = Path(job["output"])
    status_file = out / "manifest.json"
    metric_file = out / "evaluations.jsonl"
    if not status_file.is_file() or not metric_file.is_file():
        return False
    status = read_json(status_file)
    if not status:
        return False
    saved = status.get("arguments", {})
    if bool(saved.get("reserve_calibration")) != ("--reserve-calibration" in job["command"]):
        raise RuntimeError(f"Calibration split policy changed for {out}. Use a new output path.")
    # Check only scientific hyperparameters (NOT machine-local paths or micro-batch split
    # so Kaggle 2x4 -> RTX 5090 4x2 transfers resume cleanly!).
    for flag in (
        "variant", "detector", "seed", "max_side", "pseudo_threshold",
        "lr", "lr_schedule", "lr_warmup_steps",
        "box_weight", "class_weight", "confidence_weight",
        "localization_scale", "localization_enabled", "adaptive_strength",
        "condition_offset", "consistency_weight",
        "conformal_alpha_strict", "conformal_vis_weight",
        "care_tri_zone", "care_nwd",
    ):
        planned = _flag(job["command"], flag)
        if planned is not None and saved.get(flag) is not None and str(saved.get(flag)) != str(planned):
            raise RuntimeError(
                f"Run settings changed for {out}: {flag}: "
                f"{saved.get(flag)} versus {planned}. Use a new output path."
            )
    planned_bs = _flag(job["command"], "batch_size")
    planned_ga = _flag(job["command"], "grad_accum_steps") or "1"
    if planned_bs is not None and saved.get("batch_size") is not None:
        saved_eff = int(saved.get("batch_size", 1)) * int(saved.get("grad_accum_steps", 1))
        planned_eff = int(planned_bs) * int(planned_ga)
        if saved_eff != planned_eff:
            raise RuntimeError(
                f"Effective batch size changed for {out}: {saved_eff} vs {planned_eff}. Use a new output path."
            )
    if status.get("status") != "completed":
        return False
    # Verify final checkpoint exists (either rolling CheckpointManager or last.pth)
    ckpt_mgr = CheckpointManager(out / "checkpoints")
    if ckpt_mgr.latest_info() is None and not (out / "last.pth").is_file():
        return False
    rows = read_jsonl(metric_file)
    if not rows:
        return False
    return int(rows[-1].get("step", 0)) >= int(job["steps"])


def _eval_complete(job) -> bool:
    out = Path(job["output"])
    steps = int(job["steps"])
    eval_dir = out / f"eval_{steps:07d}"
    det = eval_dir / "detections.json"
    met = read_json(eval_dir / "metrics.json")
    rows = read_jsonl(out / "evaluations.jsonl")
    has_row = any(int(r.get("step", -1)) == steps for r in rows)
    if not det.is_file() or not has_row:
        return False
    if not met or met.get("complete") is not True:
        return False
    return True


def _analysis_complete(job) -> bool:
    out = Path(job["output"])
    steps = int(job["steps"])
    det = out / f"eval_{steps:07d}" / "detections.json"
    an_dir = out / "analysis"
    pc = an_dir / "per_class.json"
    cd = an_dir / "conditions.json"
    if not (det.is_file() and pc.is_file() and cd.is_file()):
        return False
    if read_json(pc) is None or read_json(cd) is None:
        return False
    manifest = read_json(an_dir / "analysis_manifest.json")
    if manifest is not None and manifest.get("detections_sha256") != sha256_file(det):
        return False
    return True


def summarize(root):
    """Only collect actual completed run metrics; never manufacture AP."""
    runs = root / "runs"
    records = []
    if runs.exists():
        for manifest in runs.rglob("manifest.json"):
            metadata = read_json(manifest)
            if not metadata:
                continue
            evaluation = manifest.parent / "evaluations.jsonl"
            if metadata.get("status") != "completed" or not evaluation.is_file():
                continue
            rows = read_jsonl(evaluation)
            if not rows:
                continue
            last = rows[-1]
            args = metadata.get("arguments", {})
            records.append({
                "experiment": str(manifest.parent.relative_to(runs)).replace("\\", "/"),
                "variant": args.get("variant"),
                "seed": args.get("seed"),
                "threshold": args.get("pseudo_threshold"),
                "box_weight": args.get("box_weight"),
                "class_weight": args.get("class_weight"),
                "confidence_weight": args.get("confidence_weight"),
                "localization_scale": args.get("localization_scale"),
                "adaptive_strength": args.get("adaptive_strength"),
                "condition_offset": args.get("condition_offset"),
                "final_step": last.get("step"),
                "AP": last.get("AP"),
                "AP50": last.get("AP50"),
                "AP75": last.get("AP75"),
                "AP_small": last.get("AP_small"),
                "AP_medium": last.get("AP_medium"),
                "AP_large": last.get("AP_large"),
                "partial_validation": last.get("partial_validation"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "gpu_peak_reserved_gib": metadata.get("gpu_peak_reserved_gib"),
            })
    records.sort(key=lambda r: r["experiment"])
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "results.csv"
    if records:
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        atomic_write_text(dest, buf.getvalue())
    else:
        atomic_write_text(dest, "experiment,variant,seed,AP,AP50,AP75\n")
    atomic_write_json(root / "results.json", records)
    return len(records)


def analyze_final(job, args):
    if job["type"] == "train" and args.preset in ("core", "care", "q1_a100", "q1_paper", "paper"):
        final_predictions = Path(job["output"]) / f"eval_{job['steps']:07d}" / "detections.json"
        if not final_predictions.is_file():
            raise RuntimeError(f"Missing final COCO detections: {final_predictions}")
        tokens = job["name"].split("/")
        split_root = args.output / "splits" / tokens[0]
        coco_val = split_root / "annotations" / "val.json"
        analysis_dir = Path(job["output"]) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for script, options in (
            ("eval_per_class.py", [
                "--coco-val", str(coco_val),
                "--predictions", str(final_predictions),
                "--output", str(analysis_dir / "per_class.json"),
            ]),
            ("eval_conditions.py", [
                "--bdd-val-json", str(args.val_labels),
                "--coco-val", str(coco_val),
                "--predictions", str(final_predictions),
                "--output", str(analysis_dir / "conditions.json"),
            ]),
        ):
            analysis = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), *options],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if analysis.returncode:
                raise RuntimeError(f"{script} failed: {analysis.stderr[-2000:]}")
            print(analysis.stdout.strip(), flush=True)
        atomic_write_json(analysis_dir / "analysis_manifest.json", {
            "detections_sha256": sha256_file(final_predictions),
            "completed_utc": utc_now(),
        })
    return True


def _resume_command_string(args) -> str:
    parts = ["python", "scripts/run_suite.py", "--preset", args.preset, "--resume"]
    if args.output != _default_output_for_preset(args.preset):
        parts += ["--output", str(args.output)]
    if args.device != "auto":
        parts += ["--device", args.device]
    if args.amp != "auto":
        parts += ["--amp", args.amp]
    if args.sync_dir:
        parts += ["--sync-dir", str(args.sync_dir)]
    if args.sync_github:
        parts.append("--sync-github")
    return " ".join(parts)


def _mirror_workspace_state(args, registry: ExperimentRegistry) -> None:
    if not args.update_workspace_docs:
        return
    ws_dir = ROOT / "work_dirs"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ws_db = ws_dir / "experiment_state.sqlite"
    if registry.db_path.resolve() != ws_db.resolve():
        with contextlib.suppress(Exception):
            backup_sqlite(registry.db_path, ws_db)
    suite_state = {
        "preset": args.preset,
        "output": str(args.output),
        "db_path": str(registry.db_path),
        "command": _resume_command_string(args),
        "args": {
            "train_labels": str(args.train_labels),
            "val_labels": str(args.val_labels),
            "train_images": str(args.train_images),
            "val_images": str(args.val_images),
            "device": args.device,
            "amp": args.amp,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "detector": args.detector,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": args.lr_warmup_steps,
            "warmup_steps": args.warmup_steps,
            "train_steps": args.train_steps,
            "max_side": args.max_side,
            "save_every": args.save_every,
            "sync_dir": str(args.sync_dir) if args.sync_dir else None,
        },
        "updated_utc": utc_now(),
    }
    atomic_write_json(ws_dir / "last_suite.json", suite_state)


def _refresh_progress(args, registry: ExperimentRegistry) -> dict:
    cmd = _resume_command_string(args)
    payload = write_progress_reports(
        registry,
        preset=args.preset,
        output_dir=args.output,
        resume_command=cmd,
        workspace_root=ROOT,
        update_workspace_docs=bool(args.update_workspace_docs),
    )
    _mirror_workspace_state(args, registry)
    return payload


def _sync_disk_state_to_registry(args, dag: PipelineDAG, registry: ExperimentRegistry) -> None:
    """Reconcile existing on-disk artifacts (e.g., from imported bundles) into SQLite."""
    for node in dag.topological_order():
        rec = registry.get_stage(node.experiment_id)
        if rec is None:
            continue
        job = node.payload.get("job")
        if node.stage_type == "dataset_splitting" and job is not None:
            try:
                if job_complete(job):
                    manifest = read_json(Path(job["output"]), {}) or {}
                    if rec["status"] != "COMPLETED":
                        registry.mark_completed(
                            node.experiment_id,
                            artifacts={"split_manifest": job["output"]},
                            split_fingerprint=manifest.get("split_fingerprint"),
                        )
                elif rec["status"] == "COMPLETED":
                    registry.invalidate_stage(node.experiment_id, "split artifacts missing or incomplete on disk")
            except RuntimeError as exc:
                registry.invalidate_stage(node.experiment_id, str(exc))
        elif node.stage_type in (
            "supervised_warmup", "baseline_training", "mean_teacher_training",
            "proposed_training", "threshold_experiment", "ablation_experiment",
        ) and job is not None:
            out = Path(job["output"])
            ckpt_mgr = CheckpointManager(out / "checkpoints")
            latest = ckpt_mgr.latest_info()
            manifest = read_json(out / "manifest.json") or {}
            try:
                done = job_complete(job)
            except RuntimeError as exc:
                registry.invalidate_stage(node.experiment_id, str(exc))
                continue
            if done:
                if rec["status"] != "COMPLETED":
                    registry.mark_completed(
                        node.experiment_id,
                        last_completed_step=int(job["steps"]),
                        checkpoint_path=(latest or {}).get("path") or str(out / "last.pth"),
                        checkpoint_sha256=(latest or {}).get("sha256"),
                        accumulated_runtime_sec=float(manifest.get("elapsed_seconds") or rec.get("accumulated_runtime_sec") or 0.0),
                        config_fingerprint=node.config_fingerprint,
                        dataset_fingerprint=node.dataset_fingerprint,
                        split_fingerprint=node.split_fingerprint,
                        resume_mode=manifest.get("resume_mode") or "exact",
                    )
            else:
                if rec["status"] == "COMPLETED":
                    registry.invalidate_stage(node.experiment_id, "training artifacts/checkpoint missing on disk")
                elif latest is not None and int(latest.get("step", 0)) > int(rec.get("last_completed_step") or 0):
                    registry.update_progress(
                        node.experiment_id,
                        last_completed_step=int(latest["step"]),
                        checkpoint_path=latest["path"],
                        checkpoint_sha256=latest.get("sha256"),
                        accumulated_runtime_sec=float(manifest.get("elapsed_seconds") or rec.get("accumulated_runtime_sec") or 0.0),
                    )
        elif node.stage_type == "model_evaluation" and job is not None:
            if _eval_complete(job):
                if rec["status"] != "COMPLETED":
                    registry.mark_completed(
                        node.experiment_id,
                        artifacts={"detections": str(Path(job["output"]) / f"eval_{job['steps']:07d}" / "detections.json")},
                    )
            elif rec["status"] == "COMPLETED":
                registry.invalidate_stage(node.experiment_id, "evaluation artifacts missing on disk")
        elif node.stage_type == "condition_per_class_evaluation" and job is not None:
            if _analysis_complete(job):
                if rec["status"] != "COMPLETED":
                    registry.mark_completed(
                        node.experiment_id,
                        artifacts={
                            "per_class": str(Path(job["output"]) / "analysis" / "per_class.json"),
                            "conditions": str(Path(job["output"]) / "analysis" / "conditions.json"),
                        },
                    )
            elif rec["status"] == "COMPLETED":
                registry.invalidate_stage(node.experiment_id, "per-class/condition analysis artifacts missing on disk")


def _print_status_report(payload: dict) -> None:
    s = payload["summary"]
    est = payload["runtime_estimates"]
    c = s["counts"]
    print(f"=== SSOD Progress Report (preset={payload['preset']}) ===", flush=True)
    print(f"Registry DB : {payload['registry_db']}", flush=True)
    print(f"Output Dir  : {payload['output_dir']}", flush=True)
    print(
        f"Stages      : {s['total_stages']} total | "
        f"{c['COMPLETED']} completed | {c['RUNNING']} running | "
        f"{c['INTERRUPTED']} interrupted | {c['FAILED']} failed | {c['PENDING']} pending",
        flush=True,
    )
    print(f"Steps       : {s['completed_steps']} / {s['total_steps']} completed", flush=True)
    print(f"Runtime     : {format_duration(s['accumulated_runtime_sec'])} accumulated", flush=True)
    print(f"ETA         : {est['estimated_remaining_human']} (source={est['measurement_source']})", flush=True)
    cur = s.get("current_stage")
    if cur:
        print(f"Current/Next: {cur['stage_name']} [{cur['status']}] step {cur['step']}/{cur['target_steps']}", flush=True)
    ckpt = s.get("last_checkpoint")
    if ckpt:
        print(f"Last Ckpt   : {ckpt['stage_name']} step {ckpt['step']} -> {ckpt['path']}", flush=True)
    print(f"Continue Cmd: {payload['resume_command']}", flush=True)


def _run_subprocess_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as out:
        out.write(f"\n[{utc_now()}] COMMAND: " + subprocess.list2cmdline(command) + "\n")
        out.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf8",
            errors="replace",
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            with contextlib.suppress(OSError):
                process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            raise


def _run_environment_stage(node: StageNode, args, registry: ExperimentRegistry) -> None:
    env_file = args.output / "environment.json"
    rec = registry.get_stage(node.experiment_id)
    if rec and rec["status"] == "COMPLETED" and env_file.is_file() and args.resume:
        return
    t0 = time.monotonic()
    registry.mark_running(node.experiment_id)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_environment.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode:
        registry.mark_failed(node.experiment_id, f"check_environment failed: {proc.stderr[-1000:]}")
        raise RuntimeError(f"check_environment failed: {proc.stderr[-1000:]}")
    info = json.loads(proc.stdout)
    info["runtime_environment"] = detect_runtime_environment(args.output)
    info["verified_utc"] = utc_now()
    atomic_write_json(env_file, info)
    registry.mark_completed(
        node.experiment_id,
        artifacts={"environment": str(env_file)},
        runtime_delta_sec=time.monotonic() - t0,
    )


def _run_dataset_validation_stage(node: StageNode, args, registry: ExperimentRegistry) -> None:
    val_file = args.output / "dataset_validation.json"
    rec = registry.get_stage(node.experiment_id)
    if rec and rec["status"] == "COMPLETED" and val_file.is_file() and args.resume:
        saved = read_json(val_file) or {}
        if saved.get("dataset_fingerprint") == node.dataset_fingerprint:
            return
    t0 = time.monotonic()
    registry.mark_running(node.experiment_id)
    try:
        validate_input_files(args)
        payload = {
            "train_labels": str(args.train_labels),
            "val_labels": str(args.val_labels),
            "train_images": str(args.train_images),
            "val_images": str(args.val_images),
            "train_labels_sha256": sha256_file(args.train_labels),
            "val_labels_sha256": sha256_file(args.val_labels),
            "dataset_fingerprint": node.dataset_fingerprint,
            "verified_utc": utc_now(),
        }
        atomic_write_json(val_file, payload)
    except Exception as exc:
        registry.mark_failed(node.experiment_id, repr(exc), runtime_delta_sec=time.monotonic() - t0)
        raise
    registry.mark_completed(
        node.experiment_id,
        artifacts={"dataset_validation": str(val_file)},
        dataset_fingerprint=node.dataset_fingerprint,
        runtime_delta_sec=time.monotonic() - t0,
    )


def _run_benchmark_stage(node: StageNode, args, registry: ExperimentRegistry) -> dict:
    from benchmark_gpu import run_benchmark
    _, _, _, _, side = experiment_plan(args.preset, args.train_steps, args.warmup_steps, args.max_side)
    t0 = time.monotonic()
    registry.mark_running(node.experiment_id)
    try:
        res = run_benchmark(
            device=args.device,
            amp=args.amp,
            max_side=side,
            batch_size=args.batch_size,
            steps=5 if args.device != "cpu" else 2,
            warmup=1,
            db_path=registry.db_path,
            output_dir=args.output,
            force=args.clean_restart,
        )
    except Exception as exc:
        registry.mark_failed(node.experiment_id, repr(exc), runtime_delta_sec=time.monotonic() - t0)
        raise
    registry.mark_completed(
        node.experiment_id,
        artifacts={"benchmark": str(args.output / "benchmark.json")},
        runtime_delta_sec=time.monotonic() - t0,
    )
    print("BENCHMARK", json.dumps(res), flush=True)
    return res


def _run_reports_and_plots(args, registry: ExperimentRegistry, dag: PipelineDAG, *, mark_final: bool = False) -> int:
    stat_id = make_experiment_id(args.preset, "statistical_analysis")
    plot_id = make_experiment_id(args.preset, "plot_generation")
    rep_id = make_experiment_id(args.preset, "final_report")

    t0 = time.monotonic()
    n = summarize(args.output)
    from report_results import write_report
    write_report(args.output / "results.csv", args.output)
    if mark_final and stat_id in dag.nodes:
        registry.mark_completed(
            stat_id,
            artifacts={
                "results_csv": str(args.output / "results.csv"),
                "results_json": str(args.output / "results.json"),
                "statistics": str(args.output / "statistics.json"),
            },
            runtime_delta_sec=time.monotonic() - t0,
        )

    if args.preset in ("core", "care", "q1_a100", "q1_paper", "paper"):
        t1 = time.monotonic()
        from plot_results import render
        fig_dir = render(args.output)
        if mark_final and plot_id in dag.nodes:
            registry.mark_completed(
                plot_id,
                artifacts={"figures_dir": str(fig_dir)},
                runtime_delta_sec=time.monotonic() - t1,
            )
    elif mark_final and plot_id in dag.nodes:
        registry.mark_completed(plot_id, artifacts={"note": "plots optional for smoke/pilot"})

    if mark_final and rep_id in dag.nodes:
        registry.mark_completed(
            rep_id,
            artifacts={
                "paper_results": str(args.output / "PAPER_RESULTS.md"),
                "progress_md": str(args.output / "PROJECT_PROGRESS.md"),
            },
        )
    _refresh_progress(args, registry)
    return n


def main(argv=None):
    args = parse_args(argv)

    # Status and dry-run MUST NEVER launch training, require locking, or mutate/invalidate prior runs.
    if args.status:
        args.output.mkdir(parents=True, exist_ok=True)
        registry = ExperimentRegistry(args.db_path)
        registry.recover_stale_running(stale_after_sec=args.stale_timeout, preset=args.preset)
        existing_stages = registry.list_stages(preset=args.preset)
        if not existing_stages:
            if args.preset == "smoke" and not args.train_labels.is_file():
                src = args.output / "synthetic_source"
                if (src / "synthetic_train_labels.json").is_file():
                    args.train_labels = src / "synthetic_train_labels.json"
                    args.val_labels = src / "synthetic_val_labels.json"
                    args.train_images = src / "images" / "train"
                    args.val_images = src / "images" / "val"
            jobs = generate_jobs(args)
            dag = build_pipeline_dag(args, jobs)
            dag.reconcile_with_registry(registry, stale_after_sec=args.stale_timeout)
            _sync_disk_state_to_registry(args, dag, registry)
        saved_suite = registry.load_suite_state() or {}
        cmd = saved_suite.get("command") or _resume_command_string(args)
        payload = write_progress_reports(
            registry,
            preset=args.preset,
            output_dir=args.output,
            resume_command=cmd,
            workspace_root=ROOT,
            update_workspace_docs=bool(args.update_workspace_docs),
        )
        _print_status_report(payload)
        return

    if args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        jobs = generate_jobs(args)
        dag = build_pipeline_dag(args, jobs)
        registry = ExperimentRegistry(args.db_path)
        spec = {
            "preset": args.preset,
            "n_jobs": len(jobs),
            "n_dag_stages": len(dag.nodes),
            "jobs": jobs,
        }
        atomic_write_json(args.output / "experiment_plan.json", spec)
        print(
            f"Plan (dry-run): {len(jobs)} primary jobs / {len(dag.nodes)} DAG stages; "
            f"preset={args.preset}. Output: {args.output}",
            flush=True,
        )
        for node in dag.topological_order():
            rec = registry.get_stage(node.experiment_id) or {}
            status = rec.get("status", "PENDING")
            deps = ",".join(dag.nodes[d].stage_name for d in node.dependencies) or "-"
            cmd = " ".join(node.payload["job"]["command"]) if "job" in node.payload else f"[{node.stage_type}]"
            print(f"[{status:11s}] {node.stage_name} (deps: {deps}) -> {cmd}", flush=True)
        return

    # Clean restart is ONLY allowed when explicitly requested by --clean-restart.
    if args.clean_restart and args.output.exists():
        print(f"WARNING: --clean-restart requested; wiping {args.output}", flush=True)
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    # Warn clearly on ephemeral cloud storage (Kaggle / Colab) without persistent sync.
    env_info = detect_runtime_environment(args.output)
    if env_info["is_ephemeral"] and not args.sync_dir:
        print(f"WARNING [CLOUD STORAGE]: {env_info['warning']}", flush=True)

    suite_lock = FileLock(args.output / ".suite.lock", purpose=f"run_suite:{args.preset}")
    suite_lock.acquire()
    lease = MachineLease(args.sync_dir, preset=args.preset) if args.sync_dir else None
    try:
        if lease is not None:
            lease.acquire(force=args.force_lease)
            if Path(args.sync_dir).is_dir() and any(Path(args.sync_dir).iterdir()):
                if not args.clean_restart:
                    restored = sync_directory(args.sync_dir, args.output, verify_sidecars=True)
                    print(f"Synced from persistent store {args.sync_dir}: {restored}", flush=True)

        if args.preset == "smoke":
            prepare_smoke_inputs(args)
        elif not args.benchmark_only:
            validate_input_files(args)

        jobs = generate_jobs(args)
        dag = build_pipeline_dag(args, jobs)
        registry = ExperimentRegistry(args.db_path)

        reconcile = dag.reconcile_with_registry(registry, stale_after_sec=args.stale_timeout)
        if reconcile["recovered_stale"]:
            print(f"Recovered stale RUNNING stages: {reconcile['recovered_stale']}", flush=True)
        if reconcile["invalidated"]:
            print(f"Dependency-invalidated stages: {reconcile['invalidated']}", flush=True)

        _sync_disk_state_to_registry(args, dag, registry)
        registry.save_suite_state(
            preset=args.preset,
            output=args.output,
            command=_resume_command_string(args),
            args_dict={
                "train_labels": str(args.train_labels),
                "val_labels": str(args.val_labels),
                "train_images": str(args.train_images),
                "val_images": str(args.val_images),
                "device": args.device,
                "amp": args.amp,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "detector": args.detector,
                "lr": args.lr,
                "lr_schedule": args.lr_schedule,
                "lr_warmup_steps": args.lr_warmup_steps,
                "warmup_steps": args.warmup_steps,
                "train_steps": args.train_steps,
                "max_side": args.max_side,
                "save_every": args.save_every,
                "sync_dir": str(args.sync_dir) if args.sync_dir else None,
            },
        )
        atomic_write_json(args.output / "experiment_plan.json", {
            "preset": args.preset,
            "n_jobs": len(jobs),
            "n_dag_stages": len(dag.nodes),
            "jobs": jobs,
        })
        _refresh_progress(args, registry)

        # Stage 1: Environment verification
        _run_environment_stage(dag.nodes[make_experiment_id(args.preset, "environment_verification")], args, registry)

        # Stage 3 (optional or benchmark-only): GPU benchmark
        if args.benchmark or args.benchmark_only:
            bench_node = dag.nodes[make_experiment_id(args.preset, "gpu_benchmarking")]
            _run_benchmark_stage(bench_node, args, registry)
            _refresh_progress(args, registry)
            if args.benchmark_only:
                print("Benchmark stage complete (no training launched).", flush=True)
                return

        # Stage 2: Dataset validation
        _run_dataset_validation_stage(dag.nodes[make_experiment_id(args.preset, "dataset_validation")], args, registry)

        print(
            f"Plan: {len(jobs)} primary jobs ({len(dag.nodes)} DAG stages); "
            f"preset={args.preset}. Output: {args.output}",
            flush=True,
        )

        def _post_stage_sync(stage_label: str) -> None:
            _refresh_progress(args, registry)
            if args.sync_dir:
                res = sync_directory(args.output, args.sync_dir, verify_sidecars=True)
                if lease is not None:
                    lease.heartbeat()
                if res["errors"]:
                    print(f"WARNING: persistent sync issues after {stage_label}: {res['errors']}", flush=True)
            if args.sync_github:
                gh = sync_results_to_github(
                    ROOT,
                    preset=args.preset,
                    output_dir=args.output,
                    message=f"results({args.preset}): completed {stage_label} [{host_id()}]",
                )
                print(f"GITHUB_SYNC: {json.dumps(gh)}", flush=True)

        primary_done = 0
        for i, job in enumerate(jobs):
            stage_id = make_experiment_id(args.preset, job["name"])
            eval_id = make_experiment_id(args.preset, f"{job['name']}/eval")
            an_id = make_experiment_id(args.preset, f"{job['name']}/analysis")

            # Check whether primary job is already verified COMPLETED
            already_complete = False
            if args.resume:
                rec = registry.get_stage(stage_id)
                if rec and rec["status"] == "COMPLETED" and job_complete(job):
                    already_complete = True
                elif job_complete(job):
                    already_complete = True

            if already_complete:
                print(f"[{i + 1}/{len(jobs)}] SKIP already complete: {job['name']}", flush=True)
                if job["type"] == "train":
                    # Ensure eval & analysis stages are also verified complete.
                    if eval_id in dag.nodes and not _eval_complete(job):
                        print(f"  Resuming incomplete evaluation for {job['name']}", flush=True)
                        cmd = list(job["command"]) + [
                            "--resume",
                            "--registry-db", str(registry.db_path),
                            "--experiment-id", stage_id,
                        ]
                        rc = _run_subprocess_logged(cmd, args.output / "logs" / (job["name"].replace("/", "__") + ".log"))
                        if rc:
                            raise SystemExit(rc)
                    if eval_id in dag.nodes:
                        registry.mark_completed(
                            eval_id,
                            artifacts={"detections": str(Path(job["output"]) / f"eval_{job['steps']:07d}" / "detections.json")},
                        )
                    if an_id in dag.nodes and not _analysis_complete(job):
                        t_an = time.monotonic()
                        registry.mark_running(an_id)
                        analyze_final(job, args)
                        registry.mark_completed(
                            an_id,
                            artifacts={
                                "per_class": str(Path(job["output"]) / "analysis" / "per_class.json"),
                                "conditions": str(Path(job["output"]) / "analysis" / "conditions.json"),
                            },
                            runtime_delta_sec=time.monotonic() - t_an,
                        )
                    elif an_id in dag.nodes:
                        registry.mark_completed(an_id)
                _run_reports_and_plots(args, registry, dag, mark_final=False)
                primary_done += 1
                if args.run_limit > 0 and primary_done >= args.run_limit:
                    print("Stopped by --run-limit; no claim of full experiment completion.", flush=True)
                    break
                continue

            print(f"[{i + 1}/{len(jobs)}] START {job['name']}", flush=True)
            log_file = args.output / "logs" / (job["name"].replace("/", "__") + ".log")
            t_job = time.monotonic()
            registry.mark_running(stage_id)

            cmd = list(job["command"])
            if job["type"] == "train":
                cmd += [
                    "--resume" if args.resume else "--no-resume",
                    "--registry-db", str(registry.db_path),
                    "--experiment-id", stage_id,
                ]
            try:
                return_code = _run_subprocess_logged(cmd, log_file)
            except KeyboardInterrupt:
                registry.mark_failed(
                    stage_id,
                    "Interrupted by user (KeyboardInterrupt)",
                    status="INTERRUPTED",
                    runtime_delta_sec=time.monotonic() - t_job,
                )
                _post_stage_sync(f"{job['name']}_interrupted")
                raise

            summarize(args.output)
            if return_code:
                rec_after = registry.get_stage(stage_id) or {}
                if rec_after.get("status") == "RUNNING":
                    registry.mark_failed(
                        stage_id,
                        f"Process exited with code {return_code}; see {log_file}",
                        status="FAILED",
                        runtime_delta_sec=time.monotonic() - t_job,
                    )
                _refresh_progress(args, registry)
                print(f"FAILED: {job['name']} (exit {return_code}). See {log_file}", file=sys.stderr, flush=True)
                raise SystemExit(return_code)

            if job["type"] == "prepare":
                manifest = read_json(Path(job["output"]), {}) or {}
                registry.mark_completed(
                    stage_id,
                    artifacts={"split_manifest": job["output"]},
                    split_fingerprint=manifest.get("split_fingerprint"),
                    runtime_delta_sec=time.monotonic() - t_job,
                )
            else:
                out_dir = Path(job["output"])
                manifest = read_json(out_dir / "manifest.json", {}) or {}
                registry.mark_completed(
                    stage_id,
                    last_completed_step=int(job["steps"]),
                    checkpoint_path=manifest.get("last_checkpoint") or str(out_dir / "last.pth"),
                    checkpoint_sha256=manifest.get("last_checkpoint_sha256"),
                    accumulated_runtime_sec=float(manifest.get("elapsed_seconds") or (time.monotonic() - t_job)),
                    config_fingerprint=dag.nodes[stage_id].config_fingerprint,
                    dataset_fingerprint=dag.nodes[stage_id].dataset_fingerprint,
                    split_fingerprint=dag.nodes[stage_id].split_fingerprint,
                    resume_mode=manifest.get("resume_mode") or "fresh",
                )
                if eval_id in dag.nodes:
                    registry.mark_completed(
                        eval_id,
                        artifacts={"detections": str(out_dir / f"eval_{job['steps']:07d}" / "detections.json")},
                    )
                if an_id in dag.nodes:
                    t_an = time.monotonic()
                    registry.mark_running(an_id)
                    try:
                        analyze_final(job, args)
                    except Exception as exc:
                        registry.mark_failed(an_id, repr(exc), runtime_delta_sec=time.monotonic() - t_an)
                        _refresh_progress(args, registry)
                        raise
                    registry.mark_completed(
                        an_id,
                        artifacts={
                            "per_class": str(out_dir / "analysis" / "per_class.json"),
                            "conditions": str(out_dir / "analysis" / "conditions.json"),
                        },
                        runtime_delta_sec=time.monotonic() - t_an,
                    )

            _run_reports_and_plots(args, registry, dag, mark_final=False)
            _post_stage_sync(job["name"])
            print(f"COMPLETE: {job['name']}", flush=True)
            primary_done += 1
            if args.run_limit > 0 and primary_done >= args.run_limit:
                print("Stopped by --run-limit; no claim of full experiment completion.", flush=True)
                break

        all_primary_done = primary_done >= len(jobs)
        n = _run_reports_and_plots(args, registry, dag, mark_final=all_primary_done)
        if all_primary_done:
            _post_stage_sync("suite_complete")
        print(f"Experiment runner finished. {n} completed run records in {args.output / 'results.csv'}", flush=True)
    finally:
        if lease is not None:
            lease.release()
        suite_lock.release()


if __name__ == "__main__":
    main()
