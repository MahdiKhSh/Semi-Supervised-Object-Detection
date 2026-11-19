"""Windows-native TorchVision/torch SSOD training (NOT a SoftTeacher reproduction).

The official MMDetection SoftTeacher remains a separate optional benchmark.
This is a compact, fully specified experiment implementation. No MMCV,
Linux, WSL or compiler toolchain is required on the primary execution path.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

from ssod_research.consistency import weighted_consistency_loss
from ssod_research.native_data import RoadCocoDataset, strong_view, image_inventory_check

BDD_CLASSES = ("background", "pedestrian", "rider", "car", "truck", "bus",
               "train", "motorcycle", "bicycle", "traffic light", "traffic sign")
VARIANTS = ("supervised", "mean_teacher", "reliability", "adaptive", "condition", "full")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(*, pretrained_backbone, max_side, min_size=None):
    """Same backbone and detector hyperparameters across all variants."""
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    size = int(min_size or max_side)
    detector = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=backbone_weights,
        num_classes=len(BDD_CLASSES), min_size=size, max_size=int(max_side),
        box_detections_per_img=100, trainable_backbone_layers=3)
    # TorchVision ignores trainable_backbone_layers without pretrained weights.
    # Fix the unfrozen-stage set after a no-download supervised warm start so
    # all variants actually train the SAME backbone layers.
    for name, parameter in detector.backbone.body.named_parameters():
        if not name.startswith(("layer2.", "layer3.", "layer4.")):
            parameter.requires_grad_(False)
    return detector


def portable_targets(batch, device):
    return [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in target.items()} for _, target in batch]


def cpu_batch(dataset, indices):
    return [dataset[i] for i in indices]


def sample_indices(size, n, seed, step, stream_id):
    # Stateless RNG: checkpoint restart resumes identical image ordering.
    rng = random.Random(seed * 10000019 + step * 1009 + stream_id * 13007)
    return [rng.randrange(size) for _ in range(n)]


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for a, b in zip(teacher.parameters(), student.parameters()):
            a.mul_(decay).add_(b.detach(), alpha=1 - decay)
        for a, b in zip(teacher.buffers(), student.buffers()):
            a.copy_(b)


def training_attributes(root):
    p = Path(root) / "annotations" / "train_attributes.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf8"))


def class_frequencies(root):
    obj = json.loads((Path(root) / "annotations" / "labeled.json").read_text(encoding="utf8"))
    counts = {i: 0 for i in range(1, len(BDD_CLASSES))}
    for a in obj.get("annotations", []):
        cls = int(a["category_id"])
        if cls in counts:
            counts[cls] += 1
    return counts


def compute_threshold(cls, base, *, adaptive, counts, strength, condition_enabled, attrs,
                      condition_offset):
    threshold = base
    if adaptive:
        largest = max(counts.values(), default=0)
        if largest:
            # Underrepresented classes get lower acceptance thresholds.
            relative = math.log1p(counts.get(cls, 0)) / math.log1p(largest)
            threshold -= strength * (1 - relative)
    if condition_enabled and attrs:
        if attrs.get("timeofday", "").lower() in ("night", "dawn/dusk") or attrs.get("weather", "").lower() in ("rainy", "snowy", "foggy"):
            threshold += condition_offset  # log and ablate; positive = stricter
    return max(0.05, min(0.99, threshold))


@torch.no_grad()
def pseudo_targets(teacher, images, *, variant, threshold, min_quality,
                   localization_scale, agreement_iou, counts, adaptive_strength,
                   attrs_list, condition_offset):
    """Teacher predicts on original and optionally hflipped view.

    Cross-view agreement is matched by class/IoU and guards against
    box regression instability. Return pseudo-targets in original image
    coordinates. Strong augmentation is applied separately afterward.
    """
    teacher.eval()
    standard = teacher(images)
    repeated = None
    if variant in ("reliability", "adaptive", "condition", "full"):
        repeated = teacher([torch.flip(im, dims=(-1,)) for im in images])
    result, stats = [], {"candidates": 0, "accepted": 0, "quality_sum": 0.0}
    for i, (image, pred) in enumerate(zip(images, standard)):
        scores = pred["scores"]
        labels = pred["labels"]
        boxes = pred["boxes"]
        uncs = torch.zeros_like(scores)
        if repeated is not None:
            # unmatched instances get uncertainty 1; matched use 1-IoU.
            uncs = torch.ones_like(scores)
            b_boxes = repeated[i]["boxes"].clone()
            width = image.shape[-1]
            if len(b_boxes):
                old_x1 = b_boxes[:, 0].clone()
                old_x2 = b_boxes[:, 2].clone()
                b_boxes[:, 0] = width - old_x2
                b_boxes[:, 2] = width - old_x1
            if len(boxes) and len(b_boxes):
                overlap = box_iou(boxes, b_boxes)
                overlap[labels[:, None] != repeated[i]["labels"][None, :]] = -1
                # Transfer the IoU matrix once instead of synchronizing CUDA per pair.
                agreement = overlap.detach().cpu().numpy()
                # Greedy unique assignment avoids one repeated box matching many.
                matches = sorted(
                    ((float(agreement[a, b]), a, b) for a in range(len(boxes))
                     for b in range(len(b_boxes)) if agreement[a, b] >= agreement_iou),
                    reverse=True)
                seen_a, seen_b = set(), set()
                for match_iou, a, b in matches:
                    if a in seen_a or b in seen_b:
                        continue
                    seen_a.add(a)
                    seen_b.add(b)
                    uncs[a] = 1 - match_iou
        quality = scores * torch.exp(-localization_scale * uncs)
        keep = []
        for j, cls in enumerate(labels.detach().cpu().tolist()):
            tau = compute_threshold(
                cls, threshold,
                adaptive=variant in ("adaptive", "full"),
                counts=counts, strength=adaptive_strength,
                condition_enabled=variant in ("condition", "full"),
                attrs=attrs_list[i] if attrs_list else {},
                condition_offset=condition_offset)
            meets = bool(torch.isfinite(scores[j]) and torch.isfinite(quality[j])
                         and scores[j] >= tau)
            if repeated is not None:
                meets = meets and bool(quality[j] >= min_quality and uncs[j] <= 1 - agreement_iou)
            if meets:
                keep.append(j)
        stats["candidates"] += len(scores)
        stats["accepted"] += len(keep)
        if keep:
            selection = torch.tensor(keep, dtype=torch.long, device=boxes.device)
            b = boxes[selection].detach()
            l = labels[selection].detach()
            s = scores[selection].detach()
            u = uncs[selection].detach()
            stats["quality_sum"] += float(quality[selection].sum().item())
        else:
            b = boxes.new_zeros((0, 4))
            l = labels.new_zeros((0,), dtype=torch.long)
            s = scores.new_zeros((0,))
            u = scores.new_zeros((0,))
        result.append({"boxes": b, "labels": l, "scores": s,
                       "uncertainty": u, "image_id": torch.tensor(i, device=b.device)})
    return result, stats


def consistency_roi_loss(model, images, pseudo_targets_list, *,
                         box_weight, class_weight, confidence_weight,
                         reliability_scale):
    """Differentiable ROI regularization using explicit FasterRCNN internals.

    TorchVision transforms/resizes *both* images and targets before feature
    extraction. Student ROI head is called at transformed pseudo boxes
    and trained to predict the same boxes/classes/scores. No teacher gradients.
    This intentionally costs an extra student backbone forward.
    """
    with torch.no_grad():
        detached = [{"boxes": t["boxes"].detach(), "labels": t["labels"].detach()}
                    for t in pseudo_targets_list]
    transformed, transformed_targets = model.transform(images, detached)
    proposals = [t["boxes"].detach() for t in transformed_targets]
    counts = [len(b) for b in proposals]
    if sum(counts) == 0:
        return None
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    box_features = model.roi_heads.box_roi_pool(features, proposals, transformed.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    logits, regression = model.roi_heads.box_predictor(box_features)
    boxes = model.roi_heads.box_coder.decode(regression, proposals)
    expected = len(BDD_CLASSES)
    if boxes.ndim != 3 or boxes.shape[1:] != (expected, 4):
        raise RuntimeError(f"unexpected ROI decode shape: {tuple(boxes.shape)}")
    gt = torch.cat(proposals, dim=0)
    labels = torch.cat([t["labels"] for t in transformed_targets], dim=0)
    rows = torch.arange(len(labels), device=labels.device)
    chosen = boxes[rows, labels]
    scores = torch.cat([t["scores"] for t in pseudo_targets_list]).detach()
    uncertainty = torch.cat([t["uncertainty"] for t in pseudo_targets_list]).detach()
    return weighted_consistency_loss(
        logits, chosen, gt, labels, scores, uncertainty,
        box_weight=box_weight, class_weight=class_weight,
        confidence_weight=confidence_weight, reliability_scale=reliability_scale)


def target_for_training(pseudo, *, image_id):
    out = {"boxes": pseudo["boxes"], "labels": pseudo["labels"],
           "image_id": torch.tensor(image_id, dtype=torch.int64, device=pseudo["boxes"].device)}
    box = out["boxes"]
    out["area"] = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])).clamp(min=0)
    out["iscrowd"] = torch.zeros(len(box), dtype=torch.long, device=box.device)
    return out


def checkpoint_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def evaluate(model, val_data, device, output_dir, *, max_images=None):
    """COCO AP on original-image coordinates using the canonical evaluator."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools required for scientific evaluation") from exc
    import contextlib
    import io
    model.eval()
    gt = COCO(str(val_data.annotation_file))
    predictions = []
    id_set = []
    started = time.monotonic()
    with torch.inference_mode():
        for i in range(len(val_data) if max_images is None else min(len(val_data), max_images)):
            img, target = val_data[i]
            sample_id = int(target["image_id"].item())
            original = gt.imgs[sample_id]
            id_set.append(sample_id)
            scaled_h, scaled_w = img.shape[-2:]
            output = model([img.to(device)])[0]
            for box, score, label in zip(output["boxes"].detach().cpu().tolist(),
                                         output["scores"].detach().cpu().tolist(),
                                         output["labels"].detach().cpu().tolist()):
                if not 1 <= int(label) < len(BDD_CLASSES):
                    continue
                x1, y1, x2, y2 = box
                sx = float(original["width"]) / scaled_w
                sy = float(original["height"]) / scaled_h
                x1, x2 = max(0, x1 * sx), min(float(original["width"]), x2 * sx)
                y1, y2 = max(0, y1 * sy), min(float(original["height"]), y2 * sy)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({"image_id": sample_id, "category_id": int(label),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "score": float(score)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detections.json").write_text(json.dumps(predictions), encoding="utf8")
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "images": len(id_set),
                "seconds": round(time.monotonic()-started, 2),
                "notice": "no predicted detections"}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = id_set
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]), "images": len(id_set),
            "seconds": round(time.monotonic()-started, 2)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TorchVision road SSOD experiment")
    p.add_argument("--data-root", default="data/bdd100k")
    p.add_argument("--train-images", default=None)
    p.add_argument("--val-images", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=None,
                   help="Warm-start model weights from a completed supervised warmup")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--pseudo-threshold", type=float, default=0.7)
    p.add_argument("--min-quality", type=float, default=0.3)
    p.add_argument("--agreement-iou", type=float, default=0.5)
    p.add_argument("--localization-scale", type=float, default=4.0)
    p.add_argument("--adaptive-strength", type=float, default=0.12)
    p.add_argument("--condition-offset", type=float, default=0.05)
    p.add_argument("--unsup-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=0.25)
    p.add_argument("--box-weight", type=float, default=1.0)
    p.add_argument("--class-weight", type=float, default=1.0)
    p.add_argument("--confidence-weight", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-max-images", type=int, default=0,
                   help="0 = entire validation set. Restricted subsets are diagnostics, NOT final AP.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", default="bf16", choices=("off", "fp16", "bf16"))
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.max_side < 64:
        p.error("steps, batch size and max_side out of range")
    for name in ("pseudo_threshold", "min_quality", "agreement_iou", "ema"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            p.error(f"{name} must be in [0,1]")
    for name in ("lr", "unsup_weight", "consistency_weight", "localization_scale",
                 "adaptive_strength", "box_weight", "class_weight", "confidence_weight"):
        if getattr(args, name) < 0:
            p.error(f"{name} must be nonnegative")
    if args.condition_offset < -0.5 or args.condition_offset > 0.5:
        p.error("condition_offset must be between -0.5 and 0.5")
    return args


def train(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: configure correct RTX 5090 PyTorch build")
    if device.type == "cpu" and args.steps > 3:
        print("WARNING: CPU FasterRCNN training is extremely slow; use GPU for real experiments")
    root = Path(args.data_root)
    label_path = root / "annotations" / "labeled.json"
    unlabeled_path = root / "annotations" / "unlabeled.json"
    val_path = root / "annotations" / "val.json"
    train_dir = Path(args.train_images) if args.train_images else root / "images" / "train"
    val_dir = Path(args.val_images) if args.val_images else root / "images" / "val"
    inventories = {
        "labeled": image_inventory_check(label_path, train_dir),
        "validation": image_inventory_check(val_path, val_dir),
    }
    if args.variant != "supervised":
        inventories["unlabeled"] = image_inventory_check(unlabeled_path, train_dir)
    labeled = RoadCocoDataset(label_path, train_dir, args.max_side, require_labels=True)
    unlabeled = (RoadCocoDataset(unlabeled_path, train_dir, args.max_side)
                 if args.variant != "supervised" else None)
    validation = RoadCocoDataset(val_path, val_dir, args.max_side)
    named = set(im["file_name"] for im in labeled.images)
    if unlabeled is not None and named.intersection(im["file_name"] for im in unlabeled.images):
        raise ValueError("labeled/unlabeled image leakage")
    if named.intersection(im["file_name"] for im in validation.images):
        raise ValueError("train/validation image leakage")
    attrs = training_attributes(root)
    if args.variant in ("condition", "full") and not attrs:
        raise ValueError("condition variant requires train_attributes.json (re-run preparation)")
    counts = class_frequencies(root)
    config = vars(args).copy()
    config["device"] = str(device)
    dataset_hashes = {p.name: sha256_file(p) for p in (label_path, val_path)}
    if unlabeled is not None:
        dataset_hashes[unlabeled_path.name] = sha256_file(unlabeled_path)
    if args.variant in ("condition", "full"):
        dataset_hashes["train_attributes.json"] = sha256_file(root / "annotations" / "train_attributes.json")
    identity = dict(config)
    for key in ("output", "resume", "steps", "eval_every", "save_every", "eval_max_images"):
        identity.pop(key, None)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"warm-start checkpoint not found: {init_path}")
        identity["init_checkpoint_sha256"] = sha256_file(init_path)
    fingerprint = hashlib.sha256(json.dumps({"settings": identity, "datasets": dataset_hashes},
                                              sort_keys=True).encode()).hexdigest()
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": config,
                "fingerprint": fingerprint, "data": inventories, "dataset_sha256": dataset_hashes,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "status": "started", "framework": "TorchVision native research implementation; not SoftTeacher reproduction"}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf8")

    model = build_model(pretrained_backbone=args.pretrained_backbone and not bool(args.init_checkpoint),
                        max_side=args.max_side).to(device)
    teacher = copy.deepcopy(model).to(device) if unlabeled is not None else None
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    autocast_enabled = device.type == "cuda" and args.amp != "off"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    checkpoint = out / "last.pth"
    start = 0
    if args.init_checkpoint and not (args.resume and checkpoint.is_file()):
        warmup = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if warmup.get("step", 0) < 1 or "student" not in warmup:
            raise ValueError("warm-start checkpoint must be from a successful supervised run")
        model.load_state_dict(warmup["student"], strict=True)
        if teacher is not None:
            teacher.load_state_dict(warmup["student"], strict=True)
        print(f"Loaded supervised warm-start: {args.init_checkpoint}")
    if args.resume and checkpoint.is_file():
        # Checkpoints are trusted only if generated by this project.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise RuntimeError("checkpoint settings or data changed; select a new output folder")
        model.load_state_dict(state["student"])
        if teacher is not None:
            teacher.load_state_dict(state["teacher"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = int(state["step"])
        print(f"Resuming at step {start}")
    step_log = out / "steps.jsonl"
    started = time.monotonic()
    try:
        for step in range(start, args.steps):
            model.train()
            selection = sample_indices(len(labeled), args.batch_size, args.seed, step, 0)
            train_batch = cpu_batch(labeled, selection)
            images = [img.to(device) for img, _ in train_batch]
            targets = portable_targets(train_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                sup_losses = model(images, targets)
                supervised = sum(sup_losses.values())
                total = supervised
            extra = {"sup": float(supervised.detach()), "unsup": 0., "stability": 0.,
                     "candidates": 0, "accepted": 0}
            if teacher is not None:
                ids = sample_indices(len(unlabeled), args.batch_size, args.seed, step, 1)
                batch_u = cpu_batch(unlabeled, ids)
                originals = [img.to(device) for img, _ in batch_u]
                attributes = [attrs.get(unlabeled.images[idx]["file_name"], {}) for idx in ids]
                pseudo, statistics = pseudo_targets(
                    teacher, originals, variant=args.variant, threshold=args.pseudo_threshold,
                    min_quality=args.min_quality, localization_scale=args.localization_scale,
                    agreement_iou=args.agreement_iou, counts=counts,
                    adaptive_strength=args.adaptive_strength, attrs_list=attributes,
                    condition_offset=args.condition_offset)
                extra.update(statistics)
                # The same pseudo boxes are transformed with each student's image.
                rng = random.Random(args.seed * 7103 + step * 61 + 3)
                augmented = []
                for img, (_, target), p in zip(originals, batch_u, pseudo):
                    flip = rng.random() < 0.5
                    brightness = rng.uniform(0.75, 1.25)
                    contrast = rng.uniform(0.75, 1.25)
                    pseudo_target = target_for_training(p, image_id=int(target["image_id"]))
                    img, trans = strong_view(img, pseudo_target, flip=flip,
                                             brightness=brightness, contrast=contrast)
                    # For regularization preserve original teacher scores and uncertainty
                    # in the same matched order, including when horizontal flip is applied.
                    trans["scores"] = p["scores"]
                    trans["uncertainty"] = p["uncertainty"]
                    augmented.append((img, trans))
                if any(len(t["boxes"]) > 0 for _, t in augmented):
                    aug_images = [item[0] for item in augmented]
                    aug_targets = [target_for_training(item[1],
                                                      image_id=int(item[1]["image_id"]))
                                   for item in augmented]
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                        unsup_losses = model(aug_images, aug_targets)
                        unsupervised = sum(unsup_losses.values())
                        total = total + args.unsup_weight * unsupervised
                        extra["unsup"] = float(unsupervised.detach())
                        if args.variant == "full" and args.consistency_weight:
                            stability = consistency_roi_loss(
                                model, aug_images, [t for _, t in augmented],
                                box_weight=args.box_weight, class_weight=args.class_weight,
                                confidence_weight=args.confidence_weight,
                                reliability_scale=args.localization_scale)
                            if stability:
                                stab_total = sum(stability.values())
                                total = total + args.consistency_weight * stab_total
                                extra["stability"] = float(stab_total.detach())
                # Empty pseudo boxes must NOT train all proposals as background.
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                ema_update(teacher, model, min(args.ema, 1 - 1 / (step + 2)))
            entry = {"step": step + 1, "loss": round(float(total.detach()), 7),
                     "seconds": round(time.monotonic()-started, 2), **extra}
            with step_log.open("a", encoding="utf8") as f:
                f.write(json.dumps(entry) + "\n")
            if step == start or (step + 1) % max(1, min(args.save_every, 25)) == 0:
                print(json.dumps(entry), flush=True)
            if (step+1) % args.save_every == 0 or step+1 == args.steps:
                checkpoint_save(checkpoint, {
                    "fingerprint": fingerprint, "step": step+1,
                    "student": model.state_dict(),
                    "teacher": teacher.state_dict() if teacher is not None else None,
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
            if (step+1) % args.eval_every == 0 or step+1 == args.steps:
                evaluated = teacher if teacher is not None else model
                metrics = evaluate(evaluated, validation, device, out / f"eval_{step+1:07d}",
                                   max_images=args.eval_max_images or None)
                metrics.update({"step": step+1, "partial_validation": args.eval_max_images > 0})
                with (out / "evaluations.jsonl").open("a", encoding="utf8") as f:
                    f.write(json.dumps(metrics)+"\n")
                print("EVAL", json.dumps(metrics), flush=True)
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "interrupted"
        metadata["exception"] = repr(exc)
        raise
    finally:
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_seconds"] = round(time.monotonic()-started, 2)
        if device.type == "cuda":
            metadata["gpu_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        (out / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf8")


if __name__ == "__main__":
    train(parse_args())
