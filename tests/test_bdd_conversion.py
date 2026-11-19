import importlib.util
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
