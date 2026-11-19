import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "eval_conditions",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_conditions.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_condition_groups_use_exact_filename_join():
    bdd = [
        {"name": "one.jpg", "attributes": {"timeofday": "night", "weather": "rainy"}},
        {"name": "two.jpg", "attributes": {"timeofday": "daytime", "weather": "clear"}},
        {"name": "unmatched.jpg", "attributes": {"timeofday": "night"}},
    ]
    coco = {"images": [{"file_name": "one.jpg", "id": 12},
                       {"file_name": "two.jpg", "id": 20}]}
    groups = module.condition_groups(bdd, coco)
    assert groups[("timeofday", "night")] == {12}
    assert groups[("weather", "rainy")] == {12}
    assert groups[("timeofday", "daytime")] == {20}
