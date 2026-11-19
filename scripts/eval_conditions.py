#!/usr/bin/env python3
"""Evaluate saved COCO detections by BDD100K weather / time / scene.

Unlike merely averaging scores, this uses actual COCO bbox AP. Requires:
  pip install pycocotools
Input pred.json must contain standard COCO detection records with image_id,
category_id, bbox xywh, and score, matching our val.json image IDs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path


def condition_groups(bdd_records, coco_gt):
    """Return {(attribute,value): set(image_ids)} using filename join."""
    id_by_name = {img["file_name"]: img["id"] for img in coco_gt["images"]}
    groups = defaultdict(set)
    for record in bdd_records:
        img_id = id_by_name.get(record["name"])
        if img_id is None:
            continue
        attrs = record.get("attributes") or {}
        for key in ("timeofday", "weather", "scene"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                groups[(key, value.strip())].add(img_id)
    return groups


def run_evaluation(ground_truth, predictions, groups, min_images=25):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install pycocotools to evaluate detections") from exc
    import contextlib
    import io
    import tempfile

    valid_ids = {im["id"] for im in ground_truth["images"]}
    image_counts = {k: len(v & valid_ids) for k, v in groups.items()}
    results = {}
    for (attribute, value), requested_ids in sorted(groups.items()):
        subset_ids = sorted(requested_ids & valid_ids)
        if len(subset_ids) < min_images:
            continue
        anns = [ann for ann in ground_truth["annotations"] if ann["image_id"] in requested_ids]
        subset = {
            "info": ground_truth.get("info", {"description": "BDD100K condition subset"}),
            "licenses": ground_truth.get("licenses", []),
            "images": [im for im in ground_truth["images"] if im["id"] in requested_ids],
            "annotations": anns,
            "categories": ground_truth["categories"],
        }
        filtered_predictions = [p for p in predictions if p["image_id"] in requested_ids]
        # COCO.loadRes has special empty-prediction behavior: report missing AP
        # instead of silently pretending empty results yield a valid measurement.
        if not filtered_predictions:
            results[f"{attribute}/{value}"] = {"images": len(subset_ids), "AP": None,
                                                "AP50": None, "reason": "no detections"}
            continue
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            gt_path = Path(td) / "gt.json"
            gt_path.write_text(json.dumps(subset), encoding="utf-8")
            coco_gt = COCO(str(gt_path))
            coco_dt = coco_gt.loadRes(filtered_predictions)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = subset_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        results[f"{attribute}/{value}"] = {
            "images": len(subset_ids), "AP": round(float(evaluator.stats[0]), 5),
            "AP50": round(float(evaluator.stats[1]), 5), "AP75": round(float(evaluator.stats[2]), 5),
        }
    return {"metrics": results, "group_sizes": {f"{k[0]}/{k[1]}": n for k, n in image_counts.items()},
            "note": "Subsets overlap across attributes; do not aggregate their AP as independent samples."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-val-json", type=Path, required=True)
    parser.add_argument("--coco-val", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-images", type=int, default=25)
    args = parser.parse_args()
    bdd = json.loads(args.bdd_val_json.read_text(encoding="utf-8"))
    coco = json.loads(args.coco_val.read_text(encoding="utf-8"))
    pred = json.loads(args.predictions.read_text(encoding="utf-8"))
    groups = condition_groups(bdd, coco)
    report = run_evaluation(coco, pred, groups, min_images=args.min_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
