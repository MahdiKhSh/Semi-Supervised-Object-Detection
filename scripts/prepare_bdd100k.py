#!/usr/bin/env python3
"""Convert BDD100K 2D detection labels and create leakage-safe SSOD splits.

Inputs: official BDD100K image-label JSON (list of records). Never reads
validation labels for training. All outputs are written atomically, with
``split_manifest.json`` written last as the crash-safe commit record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssod_research.recovery import atomic_write_json, atomic_write_text, sha256_file, stable_hash

CATEGORIES = (
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
)
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
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id,
                "category_id": CAT2ID[cls], "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
    return {
        "info": {"description": "BDD100K RGB 2D detection COCO conversion"},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": idx, "name": name} for name, idx in CAT2ID.items()],
    }


def build_splits(train_records, fraction=0.10, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    names = [r["name"] for r in train_records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate training filenames")
    ordered = sorted(train_records, key=lambda item: item["name"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    if n < 1 or n >= len(ordered):
        raise ValueError("need at least one labeled and one unlabeled image")
    labeled, unlabeled = ordered[:n], ordered[n:]
    assert {x["name"] for x in labeled}.isdisjoint({x["name"] for x in unlabeled})
    return labeled, unlabeled


def verify_split_output(split_dir: Path, *, train_labels: Path, val_labels: Path,
                        fraction: float, seed: int) -> tuple[bool, str, dict | None]:
    """Verify split manifest + all annotation files & checksums."""
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        return False, "split_manifest.json missing", None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"split_manifest.json unreadable: {exc}", None
    if float(manifest.get("label_fraction", -1)) != float(fraction):
        return False, f"fraction mismatch ({manifest.get('label_fraction')} != {fraction})", manifest
    if int(manifest.get("seed", -1)) != int(seed):
        return False, f"seed mismatch ({manifest.get('seed')} != {seed})", manifest
    if train_labels.is_file():
        if sha256_file(train_labels) != manifest.get("train_source_sha256"):
            return False, "train source labels changed", manifest
    if val_labels.is_file():
        if sha256_file(val_labels) != manifest.get("val_source_sha256"):
            return False, "val source labels changed", manifest
    ann_dir = split_dir / "annotations"
    required = ("labeled.json", "unlabeled.json", "val.json", "train_attributes.json")
    recorded = manifest.get("artifact_sha256") or {}
    for name in required:
        file_path = ann_dir / name
        if not file_path.is_file():
            return False, f"missing annotation file: {name}", manifest
        if name in recorded and sha256_file(file_path, use_cache=False) != recorded[name]:
            return False, f"corrupted annotation file: {name}", manifest
    return True, "verified", manifest


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train-labels", required=True, type=Path)
    p.add_argument("--val-labels", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
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
    attributes = {r["name"]: r.get("attributes", {}) for r in train}
    artifact_sha256 = {}
    attr_path = ann_dir / "train_attributes.json"
    atomic_write_text(attr_path, json.dumps(attributes, separators=(",", ":")))
    artifact_sha256["train_attributes.json"] = sha256_file(attr_path, use_cache=False)
    for filename, obj in outputs.items():
        dest = ann_dir / filename
        atomic_write_text(dest, json.dumps(obj, separators=(",", ":")))
        artifact_sha256[filename] = sha256_file(dest, use_cache=False)
    train_src_sha = sha256_file(args.train_labels, use_cache=False)
    val_src_sha = sha256_file(args.val_labels, use_cache=False)
    report = {
        "seed": args.seed,
        "label_fraction": args.fraction,
        "labeled_images": len(labeled),
        "unlabeled_images": len(unlabeled),
        "validation_images": len(val),
        "labeled_name_sha256": hashlib.sha256("\n".join(sorted(x["name"] for x in labeled)).encode()).hexdigest(),
        "train_source_sha256": train_src_sha,
        "val_source_sha256": val_src_sha,
        "artifact_sha256": artifact_sha256,
        "split_fingerprint": stable_hash({
            "seed": args.seed,
            "fraction": args.fraction,
            "artifact_sha256": artifact_sha256,
        }),
        "classes": list(CATEGORIES),
        "warning": "Potential consecutive-frame leakage: split by drive/video IDs if grouping metadata is available.",
    }
    atomic_write_json(args.output / "split_manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
