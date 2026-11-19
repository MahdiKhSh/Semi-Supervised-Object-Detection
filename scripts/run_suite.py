#!/usr/bin/env python3
"""Resumable paper-scale Windows-native SSOD experiment orchestrator.

Runs commands sequentially to avoid overloading one RTX 5090.
All experiment definitions are explicit and deterministic. JSONL/CSV reports
are generated from actual evaluations, NEVER estimated metrics.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["supervised", "mean_teacher", "reliability", "adaptive", "condition", "full"]
PAPER_SEEDS = (42, 123, 2026)
PAPER_FRACTIONS = (.05, .10)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("smoke", "pilot", "core", "paper"), default="paper")
    p.add_argument("--train-labels", type=Path, default=ROOT / "data/bdd100k/raw/labels_train.json")
    p.add_argument("--val-labels", type=Path, default=ROOT / "data/bdd100k/raw/labels_val.json")
    p.add_argument("--train-images", type=Path, default=ROOT / "data/bdd100k/images/train")
    p.add_argument("--val-images", type=Path, default=ROOT / "data/bdd100k/images/val")
    p.add_argument("--output", type=Path, default=ROOT / "work_dirs/paper_runs")
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-limit", type=int, default=0,
                   help="debugging: only launch first N jobs, not a full paper result")
    return p.parse_args(argv)


def experiment_plan(preset, train_steps=None, warmup_steps=None, max_side=None):
    if preset == "smoke":
        specs = ((.10, 42),)
        selected = [("supervised", .7, "control", {}),
                    ("mean_teacher", .7, "teacher", {}),
                    ("full", .7, "combined", {})]
        return specs, selected, warmup_steps or 1, train_steps or 2, max_side or 256
    if preset == "pilot":
        specs = ((.10, 42),)
        selected = [(v, .7, v, {}) for v in VARIANTS]
        selected += [("full", .5, "threshold05", {}),
                     ("full", .9, "threshold09", {}),
                     ("full", .7, "no_box", {"box_weight": 0.}),
                     ("full", .7, "no_class", {"class_weight": 0.}),
                     ("full", .7, "no_conf", {"confidence_weight": 0.})]
        return specs, selected, warmup_steps or 30, train_steps or 100, max_side or 384
    if preset == "core":
        specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
        selected = [(v, .7, v, {}) for v in
                    ("supervised", "mean_teacher", "reliability", "full")]
        selected += [
            ("full", .5, "threshold05", {}),
            ("full", .9, "threshold09", {}),
            ("full", .7, "no_box", {"box_weight": 0.}),
            ("full", .7, "no_class", {"class_weight": 0.}),
            ("full", .7, "no_conf", {"confidence_weight": 0.}),
            ("full", .7, "no_localization", {"localization_enabled": 0}),
            ("full", .7, "no_condition", {"condition_offset": 0.}),
            ("full", .7, "no_extra_consistency", {"consistency_weight": 0.}),
        ]
        return specs, selected, warmup_steps or 1000, train_steps or 3000, max_side or 512
    specs = tuple((f, s) for f in PAPER_FRACTIONS for s in PAPER_SEEDS)
    selected = [(v, .7, v, {}) for v in VARIANTS]
    selected += [
        ("full", .5, "threshold05", {}),
        ("full", .9, "threshold09", {}),
        ("full", .7, "no_box", {"box_weight": 0.}),
        ("full", .7, "no_class", {"class_weight": 0.}),
        ("full", .7, "no_conf", {"confidence_weight": 0.}),
        ("full", .7, "no_localization", {"localization_scale": 0.}),
        ("full", .7, "no_adaptive", {"adaptive_strength": 0.}),
        ("full", .7, "no_condition", {"condition_offset": 0.}),
        ("full", .7, "no_extra_consistency", {"consistency_weight": 0.}),
    ]
    return specs, selected, warmup_steps or 2000, train_steps or 8000, max_side or 640


def generate_jobs(args):
    specs, variants, warmup_steps, train_steps, side = experiment_plan(
        args.preset, args.train_steps, args.warmup_steps, args.max_side)
    if not 0 < warmup_steps < 10000000 or not 0 < train_steps < 10000000:
        raise ValueError("step budgets must be positive")
    jobs = []
    for fraction, seed in specs:
        frac_tag = f"{round(100*fraction):02d}pct"
        split = args.output / "splits" / f"{frac_tag}_s{seed}"
        root = args.output / "runs" / f"{frac_tag}_s{seed}"
        base = [
            "--data-root", str(split), "--train-images", str(args.train_images),
            "--val-images", str(args.val_images), "--device", args.device,
            "--batch-size", str(args.batch_size), "--max-side", str(side),
            "--seed", str(seed), "--save-every", "250" if args.preset == "paper" else "10",
            "--eval-every", str(train_steps) if args.preset in ("smoke", "pilot") else "2000",
            "--amp", "bf16" if args.device != "cpu" else "off",
        ]
        if args.preset == "smoke":
            base += ["--no-pretrained-backbone", "--eval-max-images", "4"]
        elif args.preset == "pilot":
            base += ["--eval-max-images", "50"]
        prepare_cmd = [
            sys.executable, str(ROOT / "scripts" / "prepare_bdd100k.py"),
            "--train-labels", str(args.train_labels), "--val-labels", str(args.val_labels),
            "--output", str(split), "--fraction", str(fraction), "--seed", str(seed)]
        jobs.append({"name": f"{frac_tag}_s{seed}/prepare", "type": "prepare",
                     "output": str(split / "split_manifest.json"), "command": prepare_cmd})
        warmup_output = root / "warmup"
        warmup_cmd = [sys.executable, "-m", "ssod_research.native_training",
                      "--variant", "supervised", "--steps", str(warmup_steps),
                      "--output", str(warmup_output), *base]
        jobs.append({"name": f"{frac_tag}_s{seed}/warmup", "type": "train",
                     "output": str(warmup_output), "steps": warmup_steps,
                     "command": warmup_cmd})
        init = warmup_output / "last.pth"
        for variant, threshold, tag, kw in variants:
            if args.preset == "core" and seed != 42 and tag not in (
                "supervised", "mean_teacher", "reliability", "full"):
                continue
            output = root / tag
            run_cmd = [sys.executable, "-m", "ssod_research.native_training",
                       "--variant", variant, "--steps", str(train_steps),
                       "--output", str(output), "--init-checkpoint", str(init),
                       "--pseudo-threshold", str(threshold), *base]
            for key, value in kw.items():
                run_cmd += ["--" + key.replace("_", "-"), str(value)]
            jobs.append({"name": f"{frac_tag}_s{seed}/{tag}", "type": "train",
                         "output": str(output), "steps": train_steps,
                         "command": run_cmd})
    return jobs


def prepare_smoke_inputs(args):
    from make_synthetic_bdd import create
    source = args.output / "synthetic_source"
    paths = create(source, seed=42)
    args.train_labels = Path(paths["train_labels"])
    args.val_labels = Path(paths["val_labels"])
    args.train_images = Path(paths["train_images"])
    args.val_images = Path(paths["val_images"])


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
    if job["type"] == "prepare":
        path = Path(job["output"])
        if not path.is_file():
            return False
        manifest = json.loads(path.read_text(encoding="utf8"))
        fraction = _flag(job["command"], "fraction")
        seed = _flag(job["command"], "seed")
        if float(manifest.get("label_fraction", -1)) != float(fraction):
            raise RuntimeError(f"Split fraction changed: {path}")
        if int(manifest.get("seed", -1)) != int(seed):
            raise RuntimeError(f"Split seed changed: {path}")
        from hashlib import sha256
        for filename, key in (("train_labels", "train_source_sha256"),
                              ("val_labels", "val_source_sha256")):
            source = Path(_flag(job["command"], filename))
            if sha256(source.read_bytes()).hexdigest() != manifest.get(key):
                raise RuntimeError(f"Split source changed: {source}; use a new output folder")
        return True
    out = Path(job["output"])
    status_file = out / "manifest.json"
    metric_file = out / "evaluations.jsonl"
    if not status_file.is_file() or not metric_file.is_file():
        return False
    status = json.loads(status_file.read_text(encoding="utf8"))
    saved = status.get("arguments", {})
    for flag in ("variant", "seed", "max_side", "batch_size", "pseudo_threshold",
                 "box_weight", "class_weight", "confidence_weight",
                 "localization_scale", "localization_enabled", "adaptive_strength", "condition_offset",
                 "consistency_weight", "init_checkpoint", "train_images", "val_images"):
        planned = _flag(job["command"], flag)
        if planned is not None and str(saved.get(flag)) != str(planned):
            raise RuntimeError(f"Run settings changed for {out}: {flag}: "
                               f"{saved.get(flag)} versus {planned}. Use a new output path.")
    if status.get("status") != "completed":
        return False
    lines = [s for s in metric_file.read_text(encoding="utf8").splitlines() if s.strip()]
    if not lines:
        return False
    last = json.loads(lines[-1])
    return int(last.get("step", 0)) >= int(job["steps"])


def summarize(root):
    """Only collect actual completed run metrics; never manufacture AP."""
    runs = root / "runs"
    records = []
    if runs.exists():
        for manifest in runs.rglob("manifest.json"):
            metadata = json.loads(manifest.read_text(encoding="utf8"))
            evaluation = manifest.parent / "evaluations.jsonl"
            if metadata.get("status") != "completed" or not evaluation.is_file():
                continue
            lines = [line for line in evaluation.read_text(encoding="utf8").splitlines() if line.strip()]
            if not lines:
                continue
            last = json.loads(lines[-1])
            args = metadata.get("arguments", {})
            records.append({
                "experiment": str(manifest.parent.relative_to(runs)).replace("\\", "/"),
                "variant": args.get("variant"), "seed": args.get("seed"),
                "threshold": args.get("pseudo_threshold"),
                "box_weight": args.get("box_weight"), "class_weight": args.get("class_weight"),
                "confidence_weight": args.get("confidence_weight"),
                "localization_scale": args.get("localization_scale"),
                "adaptive_strength": args.get("adaptive_strength"),
                "condition_offset": args.get("condition_offset"),
                "final_step": last.get("step"),
                "AP": last.get("AP"), "AP50": last.get("AP50"), "AP75": last.get("AP75"),
                "partial_validation": last.get("partial_validation"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "gpu_peak_reserved_gib": metadata.get("gpu_peak_reserved_gib"),
            })
    records.sort(key=lambda r: r["experiment"])
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "results.csv"
    if records:
        with dest.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        dest.write_text("experiment,variant,seed,AP,AP50,AP75\n", encoding="utf8")
    (root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf8")
    return len(records)



def analyze_final(job, args):
    if job["type"] == "train" and args.preset in ("core", "paper"):
        # The scientific result is the full-validation final checkpoint.
        final_predictions = Path(job["output"]) / f"eval_{job['steps']:07d}" / "detections.json"
        if not final_predictions.is_file():
            raise RuntimeError(f"Missing final COCO detections: {final_predictions}")
        # The COCO validation annotations live in this job's split root.
        tokens = job["name"].split("/")
        split_root = args.output / "splits" / tokens[0]
        coco_val = split_root / "annotations" / "val.json"
        analysis_dir = Path(job["output"]) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for script, options in (
            ("eval_per_class.py",
             ["--coco-val", str(coco_val),
              "--predictions", str(final_predictions),
              "--output", str(analysis_dir / "per_class.json")]),
            ("eval_conditions.py",
             ["--bdd-val-json", str(args.val_labels),
              "--coco-val", str(coco_val),
              "--predictions", str(final_predictions),
              "--output", str(analysis_dir / "conditions.json")]),
        ):
            analysis = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *options],
                                      cwd=ROOT, capture_output=True, text=True)
            if analysis.returncode:
                raise RuntimeError(f"{script} failed: {analysis.stderr[-2000:]}")
            print(analysis.stdout.strip(), flush=True)
    return True

def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preset == "smoke" and not args.dry_run:
        prepare_smoke_inputs(args)
    elif not args.dry_run:
        validate_input_files(args)
    jobs = generate_jobs(args)
    spec = {"preset": args.preset, "n_jobs": len(jobs), "jobs": jobs}
    (args.output / "experiment_plan.json").write_text(json.dumps(spec, indent=2), encoding="utf8")
    print(f"Plan: {len(jobs)} sequential jobs; preset={args.preset}. Output: {args.output}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["name"], " ".join(job["command"]))
        return
    completed = 0
    for i, job in enumerate(jobs):
        if job_complete(job) and args.resume:
            print(f"[{i+1}/{len(jobs)}] SKIP already complete: {job['name']}", flush=True)
            if args.preset in ("core", "paper") and job["type"] == "train":
                analyze_final(job, args)
            summarize(args.output)
            from report_results import write_report
            write_report(args.output / "results.csv", args.output)
            completed += 1
            continue
        print(f"[{i+1}/{len(jobs)}] START {job['name']}", flush=True)
        log_dir = args.output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / (job["name"].replace("/", "__") + ".log")
        with log.open("a", encoding="utf8") as out:
            out.write("\nCOMMAND: " + subprocess.list2cmdline(job["command"]) + "\n")
            out.flush()
            process = subprocess.Popen(
                job["command"], cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, encoding="utf8", errors="replace",
                text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        summarize(args.output)
        if return_code:
            print(f"FAILED: {job['name']} (exit {return_code}). See {log}", file=sys.stderr, flush=True)
            raise SystemExit(return_code)
        if args.preset in ("core", "paper") and job["type"] == "train":
            analyze_final(job, args)
        from report_results import write_report
        write_report(args.output / "results.csv", args.output)
        if args.preset in ("core", "paper"):
            from plot_results import render
            render(args.output)
        print(f"COMPLETE: {job['name']}", flush=True)
        completed += 1
        if args.run_limit > 0 and completed >= args.run_limit:
            print("Stopped by --run-limit; no claim of full experiment completion.", flush=True)
            break
    n = summarize(args.output)
    from report_results import write_report
    write_report(args.output / "results.csv", args.output)
    if args.preset in ("core", "paper"):
        from plot_results import render
        render(args.output)
    print(f"Experiment runner finished. {n} completed run records in {args.output / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
