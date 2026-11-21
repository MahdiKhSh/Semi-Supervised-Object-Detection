#!/usr/bin/env python3
"""Zero-shot cross-dataset adverse-weather evaluation (e.g., DAWN / ACDC / Foggy Cityscapes).

Evaluates a BDD100K-trained checkpoint without any fine-tuning on any COCO-formatted
external road dataset sharing the 10-class BDD taxonomy (1..10). Fully resumable via
incremental partial_predictions.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssod_research.native_data import RoadCocoDataset
from ssod_research.native_training import build_model, evaluate
from ssod_research.recovery import atomic_write_json, safe_torch_load, sha256_file


def main(argv=None):
    p = argparse.ArgumentParser(description="Resumable zero-shot external adverse-weather COCO evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--coco-ann", type=Path, required=True, help="External COCO annotation JSON (BDD categories 1..10)")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=800)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = p.parse_args(argv)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    state = safe_torch_load(args.checkpoint)
    detector_type = (state.get("config") or {}).get("detector", "fasterrcnn_v1")
    model = build_model(pretrained_backbone=False, max_side=args.max_side, detector=detector_type).to(device)
    weights = state.get("teacher") or state.get("student")
    if weights is None:
        raise ValueError("Checkpoint contains neither teacher nor student weights")
    model.load_state_dict(weights, strict=True)

    dataset = RoadCocoDataset(args.coco_ann, args.images, max_side=args.max_side)
    eval_fp = f"{sha256_file(args.checkpoint)}:{args.max_side}:{args.max_images}"
    metrics = evaluate(
        model,
        dataset,
        device,
        args.output,
        max_images=args.max_images or None,
        eval_fingerprint=eval_fp,
        resume_partial=True,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "detector": detector_type,
        "dataset": str(args.coco_ann),
        "max_side": int(args.max_side),
        "metrics": metrics,
    }
    atomic_write_json(args.output / "external_eval_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
