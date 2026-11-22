import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location("bdd", Path(__file__).resolve().parents[1] / "scripts" / "prepare_bdd100k.py")
bdd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bdd)


def _record(i):
    return {"name": f"{i:03d}.jpg", "labels": [{"category": "car",
            "box2d": {"x1": 1, "y1": 2, "x2": 20, "y2": 30}}]}


def test_deterministic_disjoint_split():
    records = [_record(i) for i in range(20)]
    a, b = bdd.build_splits(records, .1, 42)
    a2, b2 = bdd.build_splits(records, .1, 42)
    assert [x["name"] for x in a] == [x["name"] for x in a2]
    assert [x["name"] for x in b] == [x["name"] for x in b2]
    assert len(a) == 2 and len(b) == 18
    assert {x["name"] for x in a}.isdisjoint({x["name"] for x in b})


def test_coco_unlabeled_has_no_annotations():
    obj = bdd.convert([_record(1)], include_annotations=False)
    assert len(obj["images"]) == 1 and obj["annotations"] == []
    assert len(obj["categories"]) == 10
    assert obj["info"]["description"]
    assert "licenses" in obj


def test_boxes_and_duplicate_rejection():
    obj = bdd.convert([_record(1)])
    assert obj["annotations"][0]["bbox"] == [1, 2, 19, 28]
    with pytest.raises(ValueError):
        bdd.convert([_record(1), _record(1)])


def test_prepare_filters_missing_images_and_supports_nested_dirs(tmp_path):
    train_labels = tmp_path / "train.json"
    val_labels = tmp_path / "val.json"
    train_records = [_record(i) for i in range(10)]
    val_records = [_record(100 + i) for i in range(4)]
    train_labels.write_text(json.dumps(train_records), encoding="utf-8")
    val_labels.write_text(json.dumps(val_records), encoding="utf-8")

    # Put only 6 of 10 train images across nested subdirs (trainA / trainB), and 2 of 4 val images
    train_dir = tmp_path / "images" / "train"
    val_dir = tmp_path / "images" / "val"
    (train_dir / "trainA").mkdir(parents=True)
    (train_dir / "trainB").mkdir(parents=True)
    val_dir.mkdir(parents=True)
    for i in range(3):
        (train_dir / "trainA" / f"{i:03d}.jpg").write_bytes(b"fake")
    for i in range(3, 6):
        (train_dir / "trainB" / f"{i:03d}.jpg").write_bytes(b"fake")
    for i in range(2):
        (val_dir / f"{100 + i:03d}.jpg").write_bytes(b"fake")

    out_dir = tmp_path / "split"
    bdd.main([
        "--train-labels", str(train_labels),
        "--val-labels", str(val_labels),
        "--train-images", str(train_dir),
        "--val-images", str(val_dir),
        "--output", str(out_dir),
        "--fraction", "0.5",
        "--seed", "42",
    ])
    manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["labeled_images"] == 3
    assert manifest["unlabeled_images"] == 3
    assert manifest["validation_images"] == 2
    assert manifest["dropped_missing_train_images"] == 4
    assert manifest["dropped_missing_val_images"] == 2
    assert manifest["image_inventory_filtered"] is True

    ok, reason, _ = bdd.verify_split_output(
        out_dir,
        train_labels=train_labels,
        val_labels=val_labels,
        fraction=0.5,
        seed=42,
        train_images=train_dir,
        val_images=val_dir,
    )
    assert ok, reason
