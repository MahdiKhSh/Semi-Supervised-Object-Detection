#!/usr/bin/env python3
"""Convert BDD100K 2D detection labels and create leakage-safe SSOD splits.

Inputs: official BDD100K image-label JSON (list of records). Never reads
validation labels for training. Reports are descriptive, not benchmarks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

CATEGORIES = ("pedestrian", "rider", "car", "truck", "bus", "train",
              "motorcycle", "bicycle", "traffic light", "traffic sign")
CAT2ID = {name: i + 1 for i, name in enumerate(CATEGORIES)}


def convert(records, include_annotations=True):
    images, annotations, seen = [], [], set()
    for image_id, item in enumerate(records, 1):
        filename = item["name"]
        if filename in seen:
            raise ValueError(f"duplicate image name: {filename}")
        seen.add(filename)
        width, height = int(item.get("width", 1280)), int(item.get("height", 720))
        if width <= 0 or height <= 0:
            raise ValueError("invalid image size")
        images.append({"id": image_id, "file_name": filename, "width": width, "height": height})
        if not include_annotations:
            continue
        for label in item.get("labels", []):
            cls, box = label.get("category"), label.get("box2d")
            if cls not in CAT2ID or not box:
                continue
            x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
            x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": CAT2ID[cls], "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": idx, "name": name} for name, idx in CAT2ID.items()]}


def build_splits(train_records, fraction=0.10, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    names = [r["name"] for r in train_records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate training filenames")
    ordered = sorted(train_records, key=lambda item: item["name"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = max(1, min(len(ordered)-1, round(len(ordered)*fraction)))
    if n < 1 or n >= len(ordered):
        raise ValueError("need at least one labeled and one unlabeled image")
    labeled, unlabeled = ordered[:n], ordered[n:]
    assert {x["name"] for x in labeled}.isdisjoint({x["name"] for x in unlabeled})
    return labeled, unlabeled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-labels", required=True, type=Path)
    p.add_argument("--val-labels", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train = json.loads(args.train_labels.read_text(encoding="utf-8"))
    val = json.loads(args.val_labels.read_text(encoding="utf-8"))
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError("BDD labels must be JSON lists")
    names_train, names_val = {r["name"] for r in train}, {r["name"] for r in val}
    if names_train.intersection(names_val):
        raise ValueError("train and validation images overlap")
    labeled, unlabeled = build_splits(train, args.fraction, args.seed)
    ann_dir = args.output / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "labeled.json": convert(labeled, True),
        "unlabeled.json": convert(unlabeled, False),
        "val.json": convert(val, True),
    }
    for filename, obj in outputs.items():
        (ann_dir / filename).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    report = {
        "seed": args.seed, "label_fraction": args.fraction,
        "labeled_images": len(labeled), "unlabeled_images": len(unlabeled),
        "validation_images": len(val),
        "labeled_name_sha256": hashlib.sha256("\n".join(sorted(x["name"] for x in labeled)).encode()).hexdigest(),
        "train_source_sha256": hashlib.sha256(args.train_labels.read_bytes()).hexdigest(),
        "val_source_sha256": hashlib.sha256(args.val_labels.read_bytes()).hexdigest(),
        "classes": list(CATEGORIES),
        "warning": "Potential consecutive-frame leakage: split by drive/video IDs if grouping metadata is available."
    }
    (args.output / "split_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
