#!/usr/bin/env python3
"""Validate MMEngine config resolution and key research invariants, no GPU.

This is deliberately a static validation: importing the custom detector
requires torch, mmcv compiled ops and mmdet, tested separately on the GPU host.
"""
from pathlib import Path
from mmengine.config import Config

BASE = Path(__file__).resolve().parents[1]
FILES = ["bdd_supervised.py", "bdd_softteacher.py", "bdd_reliability.py"]


def inspect():
    configurations = {}
    for name in FILES:
        cfg = Config.fromfile(str(BASE / "configs" / name),
                              import_custom_modules=False)
        model = cfg.model
        klass = model.get("type")
        expected_type = {
            "bdd_supervised.py": "FasterRCNN",
            "bdd_softteacher.py": "SoftTeacher",
            "bdd_reliability.py": "ReliabilitySoftTeacher",
        }[name]
        if klass != expected_type:
            raise AssertionError(f"{name}: expected {expected_type}, got {klass}")
        if name == "bdd_supervised.py":
            num_classes = model.roi_head.bbox_head.num_classes
        else:
            num_classes = model.detector.roi_head.bbox_head.num_classes
        if num_classes != 10:
            raise AssertionError(f"{name}: num_classes={num_classes}")
        if cfg.val_dataloader.dataset.ann_file != "annotations/val.json":
            raise AssertionError("incorrect validation annotation path")
        if cfg.train_cfg.max_iters < 2:
            raise AssertionError("training iteration budget too small")
        configurations[name] = klass
    return configurations


if __name__ == "__main__":
    print(inspect())
