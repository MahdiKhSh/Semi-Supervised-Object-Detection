"""Official COCO.loadRes + per-class and weather/time evaluation integration."""
from pathlib import Path
import importlib.util
import json
import pytest

pytest.importorskip("pycocotools")


def load_script(name):
    file = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_coco_backend_nonempty_predictions(tmp_path):
    converter = load_script("prepare_bdd100k")
    perclass = load_script("eval_per_class")
    conditions = load_script("eval_conditions")
    bdd = [{"name": "scene001.jpg", "width": 320, "height": 192,
            "attributes": {"timeofday": "night", "weather": "rainy"},
            "labels": [{"category": "car", "box2d": {
                "x1": 10, "y1": 20, "x2": 80, "y2": 75}}]}]
    coco = converter.convert(bdd)
    gt_path = tmp_path / "val.json"
    gt_path.write_text(json.dumps(coco), encoding="utf8")
    pred = [{"image_id": 1, "category_id": 3, "bbox": [10, 20, 70, 55], "score": 0.99}]
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(pred), encoding="utf8")
    classes = perclass.class_metrics(gt_path, pred_path)["classes"]
    car = next(x for x in classes if x["category"] == "car")
    assert car["ground_truth_count"] == 1
    assert car["AP50"] > 0.99
    groups = conditions.condition_groups(bdd, coco)
    assert groups[("timeofday", "night")] == {1}
    report = conditions.run_evaluation(coco, pred, groups, min_images=1)
    assert report["metrics"]["timeofday/night"]["AP50"] > .99
