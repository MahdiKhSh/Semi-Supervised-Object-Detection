import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from ssod_research.native_data import RoadCocoDataset, image_inventory_check, strong_view


def test_native_dataset_resize_and_flip(tmp_path):
    from PIL import Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (200, 100), color="red").save(img_dir / "a.jpg")
    anns = tmp_path / "dataset.json"
    anns.write_text(json.dumps({
        "images": [{"id": 7, "file_name": "a.jpg", "width": 200, "height": 100}],
        "categories": [{"id": i, "name": f"c{i}"} for i in range(1, 11)],
        "annotations": [{"id": 1, "image_id": 7, "category_id": 3, "bbox": [10, 20, 50, 30]}],
    }), encoding="utf8")
    ds = RoadCocoDataset(anns, img_dir, max_side=100, require_labels=True)
    im, target = ds[0]
    assert tuple(im.shape) == (3, 50, 100)
    assert target["labels"].tolist() == [3]
    torch.testing.assert_close(target["boxes"], torch.tensor([[5., 10., 30., 25.]]))
    flipped, t = strong_view(im, target, flip=True)
    torch.testing.assert_close(t["boxes"], torch.tensor([[70., 10., 95., 25.]]))
    torch.testing.assert_close(target["boxes"], torch.tensor([[5., 10., 30., 25.]]))
    assert flipped.shape == im.shape


def test_native_dataset_nested_subdirectories(tmp_path):
    from PIL import Image
    img_dir = tmp_path / "images"
    (img_dir / "trainA").mkdir(parents=True)
    (img_dir / "trainB").mkdir(parents=True)
    Image.new("RGB", (120, 80), color="blue").save(img_dir / "trainA" / "a.jpg")
    Image.new("RGB", (120, 80), color="green").save(img_dir / "trainB" / "b.jpg")
    anns = tmp_path / "dataset.json"
    anns.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 120, "height": 80},
            {"id": 2, "file_name": "b.jpg", "width": 120, "height": 80},
        ],
        "categories": [{"id": i, "name": f"c{i}"} for i in range(1, 11)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 20]},
            {"id": 2, "image_id": 2, "category_id": 3, "bbox": [20, 15, 30, 25]},
        ],
    }), encoding="utf8")
    inv = image_inventory_check(anns, img_dir)
    assert inv["count"] == 2
    ds = RoadCocoDataset(anns, img_dir, max_side=120, require_labels=True)
    assert len(ds) == 2
    im0, _ = ds[0]
    im1, _ = ds[1]
    assert tuple(im0.shape) == (3, 80, 120)
    assert tuple(im1.shape) == (3, 80, 120)
