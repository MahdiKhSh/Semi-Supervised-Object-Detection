#!/usr/bin/env python3
"""Resumable GPU/CPU throughput benchmark for honest runtime estimation.

Runs a short synthetic FasterRCNN benchmark (never touches real BDD100K or
reports AP), caches results in the persistent SQLite experiment registry and
writes ``runtime_estimates.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssod_research.recovery import atomic_write_json, host_id, utc_now
from ssod_research.registry import ExperimentRegistry


def run_benchmark(
    *,
    device: str = "auto",
    amp: str = "auto",
    max_side: int = 512,
    batch_size: int = 1,
    steps: int = 5,
    warmup: int = 1,
    db_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict:
    import torch
    from ssod_research.native_training import build_model, resolve_amp

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for benchmark but not available")
    amp_mode, autocast_enabled, dtype = resolve_amp(amp, dev)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else None

    registry = ExperimentRegistry(db_path) if db_path else None
    if registry is not None and not force:
        cached = registry.get_benchmark(
            gpu_name=gpu_name,
            device_type=dev.type,
            amp=amp_mode,
            max_side=max_side,
            batch_size=batch_size,
        )
        if cached is not None and int(cached.get("measured_steps", 0)) >= int(steps):
            result = {
                "cached": True,
                "gpu_name": cached["gpu_name"],
                "device_type": cached["device_type"],
                "amp": cached["amp"],
                "max_side": cached["max_side"],
                "batch_size": cached["batch_size"],
                "sec_per_step": cached["sec_per_step"],
                "peak_vram_gib": cached["peak_vram_gib"],
                "measured_steps": cached["measured_steps"],
                "updated_utc": cached["updated_utc"],
                "host": host_id(),
            }
            if output_dir:
                atomic_write_json(Path(output_dir) / "benchmark.json", result)
            return result

    model = build_model(pretrained_backbone=False, max_side=max_side).to(dev)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp_mode == "fp16"))

    h = max(64, int(round(max_side * 9 / 16)))
    w = int(max_side)
    images = [torch.rand(3, h, w, device=dev) for _ in range(batch_size)]
    targets = [
        {
            "boxes": torch.tensor([[16.0, 16.0, min(w - 4.0, 96.0), min(h - 4.0, 72.0)]], device=dev),
            "labels": torch.tensor([3], dtype=torch.int64, device=dev),
            "image_id": torch.tensor(idx, dtype=torch.int64, device=dev),
            "area": torch.tensor([4480.0], device=dev),
            "iscrowd": torch.tensor([0], dtype=torch.int64, device=dev),
        }
        for idx in range(batch_size)
    ]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def _one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=autocast_enabled):
            losses = model(images, targets)
            total = sum(losses.values())
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(max(0, int(warmup))):
        _one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    t0 = time.monotonic()
    for _ in range(max(1, int(steps))):
        _one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.monotonic() - t0
    sec_per_step = round(elapsed / max(1, int(steps)), 4)
    peak_vram = round(torch.cuda.max_memory_reserved(dev) / 2**30, 3) if dev.type == "cuda" else None

    result = {
        "cached": False,
        "gpu_name": gpu_name,
        "device_type": dev.type,
        "amp": amp_mode,
        "max_side": int(max_side),
        "batch_size": int(batch_size),
        "sec_per_step": sec_per_step,
        "steps_per_hour": round(3600.0 / sec_per_step, 1) if sec_per_step > 0 else None,
        "peak_vram_gib": peak_vram,
        "measured_steps": int(steps),
        "elapsed_seconds": round(elapsed, 3),
        "updated_utc": utc_now(),
        "host": host_id(),
    }

    if registry is not None:
        registry.save_benchmark(
            gpu_name=gpu_name,
            device_type=dev.type,
            amp=amp_mode,
            max_side=max_side,
            batch_size=batch_size,
            sec_per_step=sec_per_step,
            peak_vram_gib=peak_vram,
            measured_steps=steps,
            details=result,
        )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "benchmark.json", result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Resumable SSOD GPU/CPU benchmark")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--db-path", type=Path, default=ROOT / "work_dirs" / "experiment_state.sqlite")
    p.add_argument("--output", type=Path, default=ROOT / "work_dirs")
    p.add_argument("--force", action="store_true", help="Re-measure even if a cached benchmark exists")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    res = run_benchmark(
        device=args.device,
        amp=args.amp,
        max_side=args.max_side,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        db_path=args.db_path,
        output_dir=args.output,
        force=args.force,
    )
    atomic_write_json(args.output / "runtime_estimates.json", {
        "generated_utc": utc_now(),
        "measurement_source": f"gpu_benchmark ({res.get('gpu_name') or res.get('device_type')})",
        "measured_sec_per_step": res["sec_per_step"],
        "latest_benchmark": res,
    })
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
