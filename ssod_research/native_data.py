"""Small, deterministic COCO-style RGB detection dataset for native Windows training.

Images and labels remain local, are never uploaded, and image resize transforms
xyxy boxes consistently. Coordinates are half-open pixel boxes after resize.
TorchVision uses background=0 and the ten BDD categories=1..10.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class RoadCocoDataset:
    def __init__(self, annotations, image_dir, max_side=640, require_labels=False):
        self.annotation_file = Path(annotations).resolve()
        self.image_dir = Path(image_dir).resolve()
        if not self.annotation_file.is_file() or not self.image_dir.is_dir():
            raise FileNotFoundError(f"missing COCO labels or image directory: {annotations}, {image_dir}")
        obj = json.loads(self.annotation_file.read_text(encoding="utf8"))
        self.images = sorted(obj["images"], key=lambda x: x["file_name"])
        if not self.images:
            raise ValueError("empty COCO image list")
        self.category_ids = {int(c["id"]) for c in obj["categories"]}
        if self.category_ids != set(range(1, 11)):
            raise ValueError("expected 10 BDD category IDs 1..10, got " + repr(sorted(self.category_ids)))
        self.annotations = defaultdict(list)
        for ann in obj.get("annotations", []):
            self.annotations[int(ann["image_id"])].append(ann)
        self.max_side = int(max_side)
        if self.max_side < 64:
            raise ValueError("max_side must be >= 64")
        self.require_labels = bool(require_labels)
        if self.require_labels and not obj.get("annotations"):
            raise ValueError(f"labeled dataset contains no annotations: {annotations}")
        self._names = [im["file_name"] for im in self.images]
        if len(self._names) != len(set(self._names)):
            raise ValueError("duplicate filenames in COCO dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        path = self.image_dir / info["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as pil:
            image = pil.convert("RGB")
        w, h = image.size
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image dimensions: {path}")
        scale = min(1.0, self.max_side / max(w, h))
        out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
        if (out_w, out_h) != (w, h):
            image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
        sx, sy = out_w / w, out_h / h
        boxes, labels, areas, crowd = [], [], [], []
        for ann in self.annotations.get(int(info["id"]), []):
            if int(ann["category_id"]) not in self.category_ids:
                continue
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0.0, float(x) * sx), max(0.0, float(y) * sy)
            x2, y2 = min(float(out_w), (float(x) + float(bw)) * sx), min(float(out_h), (float(y) + float(bh)) * sy)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
            areas.append((x2 - x1) * (y2 - y1))
            crowd.append(int(ann.get("iscrowd", 0)))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(info["id"]), dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64),
        }
        return tensor, target


def strong_view(image, target, *, flip=False, brightness=1.0, contrast=1.0):
    """Color transforms and exact horizontal flip; never mutates original targets."""
    image = TF.adjust_brightness(image, float(brightness))
    image = TF.adjust_contrast(image, float(contrast))
    image = image.clamp(0, 1)
    boxes = target["boxes"].clone()
    if flip:
        image = TF.hflip(image)
        w = image.shape[-1]
        if len(boxes):
            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()
            boxes[:, 0], boxes[:, 2] = w - old_x2, w - old_x1
    transformed = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
    transformed["boxes"] = boxes
    return image, transformed


def image_inventory_check(annotation_file, image_dir, *, check_every_image=True):
    """Fail closed on missing files or empty GT before a multi-hour training run."""
    obj = json.loads(Path(annotation_file).read_text(encoding="utf8"))
    names = [im["file_name"] for im in obj["images"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image filenames")
    candidates = names if check_every_image else names[:50]
    missing = [name for name in candidates if not (Path(image_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images, first examples: {missing[:5]}")
    return {"count": len(names), "annotations": len(obj.get("annotations", [])), "image_directory": str(image_dir)}
