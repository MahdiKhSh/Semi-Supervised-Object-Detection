#!/usr/bin/env python3
"""Create tiny artificial BDD-format data for smoke tests only.

These data must NEVER be mixed with real BDD100K or reported as experiments.
"""
import argparse
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw


def create(output, n_train=24, n_val=6, seed=42):
    root = Path(output)
    rng = random.Random(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        labels = []
        imgdir = root / "images" / split
        imgdir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"synthetic_{split}_{i:04}.jpg"
            img = Image.new("RGB", (320, 192), (rng.randrange(15, 80), 85, 95))
            draw = ImageDraw.Draw(img)
            x1 = 30 + (i % 7) * 10
            y1 = 40 + (i % 3) * 8
            x2, y2 = x1 + 80, y1 + 55
            draw.rectangle((x1, y1, x2, y2), fill=(190, 65, 40), outline=(245, 220, 60), width=2)
            img.save(imgdir / name, quality=90)
            labels.append({
                "name": name, "width": 320, "height": 192,
                "attributes": {"timeofday": "daytime" if i % 2 == 0 else "night",
                               "weather": "clear" if i % 3 else "rainy",
                               "scene": "city street"},
                "labels": [{"category": "car", "box2d": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2}}]})
        (root / f"synthetic_{split}_labels.json").write_text(
            json.dumps(labels, indent=2), encoding="utf8")
    return {
        "train_labels": str(root / "synthetic_train_labels.json"),
        "val_labels": str(root / "synthetic_val_labels.json"),
        "train_images": str(root / "images" / "train"),
        "val_images": str(root / "images" / "val"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="work_dirs/synthetic_bdd")
    args = p.parse_args()
    print(json.dumps(create(args.output), indent=2))
