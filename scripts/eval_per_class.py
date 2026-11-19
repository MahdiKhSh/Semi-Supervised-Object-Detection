#!/usr/bin/env python3
"""Per-class official COCO AP (not informal precision or F1 proxies)."""
import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def class_metrics(ground_truth_file, predictions_file):
    gt = COCO(str(ground_truth_file))
    prediction = json.loads(Path(predictions_file).read_text(encoding="utf8"))
    if not prediction:
        return {"error": "no detections", "classes": []}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(prediction)
    results = []
    for cat_id in sorted(gt.cats):
        cat = gt.cats[cat_id]
        num_gt = sum(int(a["category_id"]) == int(cat_id) for a in gt.anns.values())
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, dt, "bbox")
            ev.params.catIds = [cat_id]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        results.append({
            "category_id": int(cat_id), "category": cat["name"],
            "ground_truth_count": num_gt,
            "AP": float(ev.stats[0]) if num_gt else None,
            "AP50": float(ev.stats[1]) if num_gt else None,
            "AP75": float(ev.stats[2]) if num_gt else None,
        })
    return {"classes": results, "note": "AP is a dataset-level COCO measure; not per-box accuracy."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-val", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = class_metrics(args.coco_val, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf8")
    print("Saved per-class AP:", args.output)


if __name__ == "__main__":
    main()
