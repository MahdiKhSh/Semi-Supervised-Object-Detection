#!/usr/bin/env python3
"""Controlled Diagnostic Experiment Matrix Runner (Section 11 — CoRe-SSOD v2.2 Audit).

Executes or dry-runs the 9 controlled diagnostic experiments required before launching
the multi-seed 48-run Q1 paper matrix, reusing a SINGLE shared 10% seed-42 split so
all code-version, calibration, batch-size, LR, anchor, and step-budget effects are
strictly isolated without dataset confounding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssod_research.recovery import atomic_write_json

EXPERIMENTS = {
    "E1_shared_warmup_reserved": {
        "purpose": "Shared burn-in reserving D_cal (6,744 train / 256 cal) at eff_batch=8 (documented protocol)",
        "stage": "warmup",
        "args": [
            "--variant", "supervised", "--detector", "fasterrcnn_v2",
            "--steps", "1500", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_constant",
            "--lr-warmup-steps", "200", "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E2_shared_warmup_unreserved": {
        "purpose": "Calibration reservation ablation: burn-in on all 7,000 labeled images (quantifies -256 image delta)",
        "stage": "warmup_unreserved",
        "args": [
            "--variant", "supervised", "--detector", "fasterrcnn_v2",
            "--steps", "1500", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_constant",
            "--lr-warmup-steps", "200",
        ],
    },
    "E3_supervised_eb8": {
        "purpose": "Matched supervised baseline at eff_batch=8, lr=0.01 (3,000 steps = 24k exposures)",
        "init_from": "E1_shared_warmup_reserved",
        "args": [
            "--variant", "supervised", "--detector", "fasterrcnn_v2",
            "--steps", "3000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "250", "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E4_core_v22_eb8_alpha010": {
        "purpose": "Proposed CoRe-SSOD v2.2 reference at eff_batch=8, lr=0.01, alpha_strict=0.10",
        "init_from": "E1_shared_warmup_reserved",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2",
            "--steps", "3000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "250", "--ema", "0.996", "--unsup-warmup-steps", "300",
            "--conformal-every", "300", "--conformal-alpha-strict", "0.10",
            "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E5_core_v22_alpha008": {
        "purpose": "Conformal risk/coverage trade-off: alpha_strict=0.08 vs 0.10",
        "init_from": "E1_shared_warmup_reserved",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2",
            "--steps", "3000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "250", "--ema", "0.996", "--unsup-warmup-steps", "300",
            "--conformal-every", "300", "--conformal-alpha-strict", "0.08",
            "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E6_core_v22_no_zerobox_filter": {
        "purpose": "Zero-pseudo-box filtering ablation (--zero-box-filter 0 vs 1)",
        "init_from": "E1_shared_warmup_reserved",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2",
            "--steps", "3000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "250", "--ema", "0.996", "--unsup-warmup-steps", "300",
            "--conformal-every", "300", "--zero-box-filter", "0",
            "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E7_bs32_lr_scaled_matched_steps": {
        "purpose": "Batch-size & linear LR scaling test: eff_batch=32 with lr=0.02 (fixed 3,000 steps = 96k exposures)",
        "stage": "bs32_lr002",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2",
            "--steps", "3000", "--batch-size", "32", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.02", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "300", "--ema", "0.996", "--unsup-warmup-steps", "300",
            "--conformal-every", "300", "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E8_small_object_anchors_and_res1024": {
        "purpose": "Small-object recall bottleneck diagnostic: road_small FPN anchors (12..384) + max_side=1024",
        "stage": "small_obj_diag",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2", "--anchor-profile", "road_small",
            "--steps", "3000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "1024", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "250", "--ema", "0.996", "--unsup-warmup-steps", "300",
            "--conformal-every", "300", "--reserve-calibration", "--stratify-calibration",
        ],
    },
    "E9_extended_training_12k": {
        "purpose": "Extended convergence check: 12,000 SSOD steps (ema=0.9985) matching full Q1 paper horizon",
        "init_from": "E1_shared_warmup_reserved",
        "args": [
            "--variant", "core_ssod", "--detector", "fasterrcnn_v2",
            "--steps", "12000", "--batch-size", "8", "--grad-accum-steps", "1",
            "--max-side", "800", "--lr", "0.01", "--lr-schedule", "warmup_cosine",
            "--lr-warmup-steps", "500", "--ema", "0.9985", "--unsup-warmup-steps", "1000",
            "--conformal-every", "500", "--reserve-calibration", "--stratify-calibration",
        ],
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Controlled Section-11 diagnostic matrix runner")
    p.add_argument("--split-dir", type=Path, required=True, help="Prepared 10pct_s42 split directory")
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "work_dirs" / "v22_diagnostics")
    p.add_argument("--eval-max-images", type=int, default=1000,
                   help="1000 for fast stage-1 triage; 0 for full 10,000-image validation promotion")
    p.add_argument("--only", nargs="*", choices=tuple(EXPERIMENTS.keys()), default=None)
    p.add_argument("--amp", default="bf16", choices=("auto", "off", "fp16", "bf16"))
    p.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selected = args.only or list(EXPERIMENTS.keys())
    args.output.mkdir(parents=True, exist_ok=True)

    plan = []
    for key in selected:
        spec = EXPERIMENTS[key]
        run_dir = args.output / key
        cmd = [
            sys.executable, "-m", "ssod_research.native_training",
            "--data-root", str(args.split_dir),
            "--train-images", str(args.train_images),
            "--val-images", str(args.val_images),
            "--output", str(run_dir),
            "--device", args.device,
            "--amp", args.amp,
            "--save-every", "300",
            "--eval-every", "1500",
            "--eval-max-images", str(args.eval_max_images),
            "--seed", "42",
            "--resume",
            *spec["args"],
        ]
        if "init_from" in spec:
            init_ckpt = args.output / spec["init_from"] / "last.pth"
            cmd += ["--init-checkpoint", str(init_ckpt)]
        plan.append({"id": key, "purpose": spec["purpose"], "output": str(run_dir), "command": cmd})

    atomic_write_json(args.output / "diagnostic_matrix_plan.json", {"experiments": plan})
    for idx, item in enumerate(plan, 1):
        print(f"[{idx}/{len(plan)}] {item['id']}: {item['purpose']}", flush=True)
        print("  CMD:", " ".join(item["command"]), flush=True)
        if not args.dry_run:
            rc = subprocess.call(item["command"], cwd=ROOT)
            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()
