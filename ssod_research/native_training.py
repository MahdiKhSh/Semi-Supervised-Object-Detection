"""Portable, crash-safe TorchVision/torch SSOD training (NOT a SoftTeacher reproduction).

Runs identically on Windows RTX 5090, Linux L40s/A100 servers, Kaggle (T4/P100),
Google Colab, and CPU CI.  No MMCV, Linux-only extensions or compiler toolchain
are required on the primary execution path.

Crash-safety guarantees:
- Exclusive OS file lock prevents concurrent writes to the same run directory.
- Rolling atomic checkpoints with SHA-256 sidecars keep at least two valid
  checkpoints and automatically fall back past corrupted files.
- Full state (student, teacher/EMA, optimizer, LR scheduler, GradScaler,
  Python/NumPy/Torch/CUDA RNG, sampler metadata, mutable algorithm state) is
  saved and verified on resume, explicitly distinguishing exact vs approximate
  recovery (e.g., cross-GPU Kaggle -> RTX 5090 migration).
- Evaluation writes incremental partial predictions so interrupted evaluations
  resume without re-running completed images or training steps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import FrozenBatchNorm2d, box_iou

from ssod_research.baselines import (
    DynamicGMMThreshold,
    soft_teacher_extract,
    soft_teacher_unsupervised_loss,
)
from ssod_research.care_ssod import (
    OnlineVisibilityClassCalibrator,
    care_unsupervised_loss,
    extract_teacher_tri_zone_targets,
    strong_view_care,
)
from ssod_research.conformal import (
    WeightedConformalRiskCalibrator,
    split_calibration_indices,
)
from ssod_research.consistency import weighted_consistency_loss
from ssod_research.native_data import RoadCocoDataset, image_inventory_check, strong_view
from ssod_research.recovery import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    FileLock,
    IncompatibleCheckpoint,
    append_jsonl,
    append_jsonl_batch,
    atomic_copy,
    atomic_write_json,
    capture_rng_state,
    host_id,
    quarantine,
    read_json,
    read_jsonl,
    restore_rng_state,
    safe_torch_load,
    sha256_file as _sha256_file,
    stable_hash,
    torch_save_atomic,
    truncate_jsonl,
    utc_now,
    verify_file,
)

BDD_CLASSES = (
    "background", "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
)
VARIANTS = (
    "supervised",
    "mean_teacher",
    "soft_teacher",
    "consistent_teacher",
    "reliability",
    "adaptive",
    "condition",
    "full",
    "care_ssod",
    "core_ssod",
)

_NON_IDENTITY_KEYS = frozenset({
    "output", "data_root", "train_images", "val_images", "init_checkpoint",
    "resume", "steps", "eval_every", "save_every", "eval_max_images",
    "device", "amp", "registry_db", "experiment_id", "keep_checkpoints",
    "stop_after_steps", "deterministic", "batch_size", "grad_accum_steps",
})


def sha256_file(path):
    return _sha256_file(path, use_cache=False)


def source_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": head, "tracked_changes": bool(dirty.strip())}


def seed_all(seed, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_amp(amp: str, device: torch.device) -> tuple[str, bool, torch.dtype | None]:
    """Select a safe autocast mode for RTX 5090, L40s, Kaggle T4/P100, Colab or CPU."""
    if device.type != "cuda" or amp == "off":
        return "off", False, None
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    major = torch.cuda.get_device_capability(device)[0]
    if amp == "auto":
        if bf16_ok:
            return "bf16", True, torch.bfloat16
        if major >= 7:
            return "fp16", True, torch.float16
        return "off", False, None
    if amp == "bf16":
        if bf16_ok:
            return "bf16", True, torch.bfloat16
        if major >= 7:
            print("WARNING: GPU does not support bf16; falling back to fp16 autocast", flush=True)
            return "fp16", True, torch.float16
        print("WARNING: GPU does not support bf16/fp16 tensor cores; falling back to fp32", flush=True)
        return "off", False, None
    return "fp16", True, torch.float16


def _replace_bn_with_frozen_bn(module: torch.nn.Module) -> int:
    """Recursively convert nn.BatchNorm2d -> FrozenBatchNorm2d preserving pretrained stats."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            fbn = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                if child.weight is not None:
                    fbn.weight.copy_(child.weight.detach())
                if child.bias is not None:
                    fbn.bias.copy_(child.bias.detach())
                if child.running_mean is not None:
                    fbn.running_mean.copy_(child.running_mean.detach())
                if child.running_var is not None:
                    fbn.running_var.copy_(child.running_var.detach())
            setattr(module, name, fbn)
            replaced += 1
        else:
            replaced += _replace_bn_with_frozen_bn(child)
    return replaced


def _replace_bn_with_group_norm(module: torch.nn.Module, max_groups: int = 32) -> int:
    """Recursively convert nn.BatchNorm2d -> nn.GroupNorm (batch-size invariant, Wu & He ECCV'18)."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            n_ch = int(child.num_features)
            n_grp = math.gcd(int(max_groups), n_ch)
            gn = torch.nn.GroupNorm(num_groups=max(1, n_grp), num_channels=n_ch, eps=child.eps, affine=child.affine)
            with torch.no_grad():
                if child.affine and child.weight is not None and child.bias is not None:
                    gn.weight.copy_(child.weight.detach())
                    gn.bias.copy_(child.bias.detach())
            setattr(module, name, gn)
            replaced += 1
        else:
            replaced += _replace_bn_with_group_norm(child, max_groups=max_groups)
    return replaced


def _convert_v2_norm_layers(model: torch.nn.Module) -> tuple[int, int]:
    """Eliminate micro-batch BN corruption & EMA-buffer mismatch in FasterRCNN-v2.

    - Backbone body (pretrained ImageNet-V2 ResNet50): BatchNorm2d -> FrozenBatchNorm2d
      preserving exact pretrained running statistics (matching v1/Detectron2 best practice).
    - FPN, RPN, and RoI heads (randomly initialized): BatchNorm2d -> GroupNorm(32, C)
      making micro-batch sizes (Kaggle 2x4 vs RTX 5090 4x2) mathematically invariant.
    """
    n_fbn = _replace_bn_with_frozen_bn(model.backbone.body)
    n_gn = 0
    if hasattr(model.backbone, "fpn"):
        n_gn += _replace_bn_with_group_norm(model.backbone.fpn)
    if hasattr(model, "rpn"):
        n_gn += _replace_bn_with_group_norm(model.rpn)
    if hasattr(model, "roi_heads"):
        n_gn += _replace_bn_with_group_norm(model.roi_heads)
    return n_fbn, n_gn


def _build_road_small_anchor_generator():
    """Multi-scale sub-32px FPN anchor generator tailored to BDD100K small road objects.

    Over 55% of BDD100K objects have min(w, h) < 16 px after resizing 1280x720 -> 800x450,
    where TorchVision's default 32x32 P2 anchor yields max IoU < 0.12. Adding 2 scales per
    FPN octave starting at 12px on P2 raises small-object anchor IoU 4x across ALL variants.
    """
    from torchvision.models.detection.rpn import AnchorGenerator

    return AnchorGenerator(
        sizes=((12, 20), (28, 44), (60, 92), (124, 188), (252, 384)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )


def build_model(
    *,
    pretrained_backbone,
    max_side,
    min_size=None,
    detector="fasterrcnn_v1",
    anchor_profile="default",
):
    """Same backbone and detector hyperparameters across all variants."""
    if anchor_profile not in ("default", "road_small"):
        raise ValueError(f"Unknown anchor_profile: {anchor_profile}")
    size = int(min_size or max_side)

    if detector == "fasterrcnn_v2":
        backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        extra_kw = {"trainable_backbone_layers": 3} if backbone_weights is not None else {}
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=backbone_weights,
            num_classes=len(BDD_CLASSES), min_size=size, max_size=int(max_side),
            box_detections_per_img=100, **extra_kw)
        if anchor_profile == "road_small":
            from torchvision.models.detection.rpn import RPNHead

            anch = _build_road_small_anchor_generator()
            model.rpn.anchor_generator = anch
            model.rpn.head = RPNHead(
                model.backbone.out_channels, anch.num_anchors_per_location()[0], conv_depth=2
            )
        _convert_v2_norm_layers(model)
    else:
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        extra_kw = {"trainable_backbone_layers": 3} if backbone_weights is not None else {}
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=backbone_weights,
            num_classes=len(BDD_CLASSES), min_size=size, max_size=int(max_side),
            box_detections_per_img=100, **extra_kw)
        if anchor_profile == "road_small":
            from torchvision.models.detection.rpn import RPNHead

            anch = _build_road_small_anchor_generator()
            model.rpn.anchor_generator = anch
            model.rpn.head = RPNHead(
                model.backbone.out_channels, anch.num_anchors_per_location()[0], conv_depth=1
            )
    # TorchVision ignores trainable_backbone_layers without pretrained weights.
    # Fix the unfrozen-stage set after a no-download supervised warm start so
    # all variants actually train the SAME backbone layers.
    for name, parameter in model.backbone.body.named_parameters():
        if not name.startswith(("layer2.", "layer3.", "layer4.")):
            parameter.requires_grad_(False)
    return model


def build_scheduler(optimizer, *, schedule: str, total_steps: int, warmup_steps: int = 0):
    total = max(1, int(total_steps))
    wu = max(0, min(int(warmup_steps), total - 1)) if total > 1 else 0
    base_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    eta_min = base_lr * 0.01
    if schedule == "warmup_constant" and wu > 0:
        w = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=wu)
        c = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[w, c], milestones=[wu])
    if schedule == "warmup_cosine" and wu > 0:
        w = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=wu)
        c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total - wu), eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[w, c], milestones=[wu])
    if schedule == "warmup_multistep" and wu > 0:
        w = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=wu)
        rem = max(1, total - wu)
        m1 = max(1, int(round(0.65 * rem)))
        m2 = max(m1 + 1, int(round(0.85 * rem)))
        ms = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[w, ms], milestones=[wu])
    if schedule in ("cosine", "warmup_cosine"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=eta_min)
    if schedule in ("multistep", "warmup_multistep"):
        m1 = max(1, int(round(0.65 * total)))
        m2 = max(m1 + 1, int(round(0.85 * total)))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)


def _weak_flip_sample(img: torch.Tensor, target: dict, flip: bool):
    """Exact weak horizontal flip for supervised training batches."""
    if not flip:
        return img, target
    flipped_img = torch.flip(img, dims=(-1,))
    w = int(flipped_img.shape[-1])
    boxes = target["boxes"].clone()
    if len(boxes):
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - old_x2
        boxes[:, 2] = w - old_x1
    flipped_target = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
    flipped_target["boxes"] = boxes
    return flipped_img, flipped_target


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
    """EMAN (Cai et al., CVPR 2021): EMA both parameters and floating-point buffers."""
    with torch.no_grad():
        for a, b in zip(teacher.parameters(), student.parameters()):
            a.mul_(decay).add_(b.detach(), alpha=1.0 - decay)
        for a, b in zip(teacher.buffers(), student.buffers()):
            if a.is_floating_point():
                if not torch.equal(a, b):
                    a.mul_(decay).add_(b.detach(), alpha=1.0 - decay)
            else:
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
                   localization_enabled,
                   attrs_list, condition_offset):
    """Teacher predicts on original and optionally hflipped view."""
    teacher.eval()
    standard = teacher(images)
    repeated = None
    if localization_enabled and variant in ("reliability", "adaptive", "condition", "full"):
        repeated = teacher([torch.flip(im, dims=(-1,)) for im in images])
    result, stats = [], {"candidates": 0, "accepted": 0, "quality_sum": 0.0}
    for i, (image, pred) in enumerate(zip(images, standard)):
        scores = pred["scores"]
        labels = pred["labels"]
        boxes = pred["boxes"]
        uncs = torch.zeros_like(scores)
        if repeated is not None:
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
                agreement = overlap.detach().cpu().numpy()
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
    """Differentiable ROI regularization using explicit FasterRCNN internals."""
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
        confidence_weight=confidence_weight, reliability_scale=reliability_scale,
        background_index=0)


def target_for_training(pseudo, *, image_id):
    out = {"boxes": pseudo["boxes"], "labels": pseudo["labels"],
           "image_id": torch.tensor(image_id, dtype=torch.int64, device=pseudo["boxes"].device)}
    box = out["boxes"]
    out["area"] = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])).clamp(min=0)
    out["iscrowd"] = torch.zeros(len(box), dtype=torch.long, device=box.device)
    return out


def checkpoint_save(path, state, sidecar_extra: dict | None = None):
    """Atomic, fsynced checkpoint save with SHA-256 sidecar."""
    return torch_save_atomic(state, path, sidecar_extra)


def evaluate(
    model,
    val_data,
    device,
    output_dir,
    *,
    max_images=None,
    eval_fingerprint: str | None = None,
    resume_partial: bool = True,
    stop_after_images: int | None = None,
    amp_dtype: torch.dtype | None = None,
    batch_size: int | None = None,
):
    """Resumable COCO AP on original-image coordinates using pycocotools.

    Persists incremental per-image detections to ``partial_predictions.jsonl``
    so an interrupted evaluation resumes from the remaining images without
    re-running completed inference.  Writes ``detections.json`` and
    ``metrics.json`` atomically once complete.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools required for scientific evaluation") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    detections_path = output_dir / "detections.json"
    partial_path = output_dir / "partial_predictions.jsonl"

    total_n = len(val_data) if max_images is None else min(len(val_data), int(max_images))
    val_hash = _sha256_file(val_data.annotation_file)
    expected_key = stable_hash({
        "val_sha256": val_hash,
        "total_images": total_n,
        "eval_fingerprint": eval_fingerprint or "",
    })

    if resume_partial and metrics_path.is_file() and detections_path.is_file():
        cached = read_json(metrics_path)
        if cached and cached.get("complete") is True and cached.get("eval_key") == expected_key:
            return cached

    model.eval()
    gt = COCO(str(val_data.annotation_file))

    completed_by_id: dict[int, list[dict]] = {}
    if resume_partial and partial_path.is_file():
        for row in read_jsonl(partial_path):
            if row.get("eval_key") == expected_key and "image_id" in row:
                completed_by_id[int(row["image_id"])] = list(row.get("preds") or [])

    id_set = [int(val_data.images[i]["id"]) for i in range(total_n)]
    pending_indices = [i for i in range(total_n) if int(val_data.images[i]["id"]) not in completed_by_id]

    bs = int(batch_size) if batch_size is not None else (8 if device.type == "cuda" else 1)
    bs = max(1, bs)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
        if (device.type == "cuda" and amp_dtype is not None)
        else contextlib.nullcontext()
    )
    n_workers = min(6, max(2, os.cpu_count() or 4)) if (device.type == "cuda" and bs > 1) else 0
    eval_pool = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 0 else None

    started = time.monotonic()
    processed_this_call = 0
    pos = 0
    try:
        with torch.inference_mode():
            while pos < len(pending_indices):
                if stop_after_images is not None and processed_this_call >= int(stop_after_images):
                    raise InterruptedError(f"Evaluation interrupted after {processed_this_call} new images")
                take = bs
                if stop_after_images is not None:
                    take = min(take, max(1, int(stop_after_images) - processed_this_call))
                chunk = pending_indices[pos:pos + take]
                pos += len(chunk)

                if eval_pool is not None and len(chunk) > 1:
                    samples = list(eval_pool.map(val_data.__getitem__, chunk))
                else:
                    samples = [val_data[i] for i in chunk]

                imgs_dev = [im.to(device) for im, _ in samples]
                with autocast_ctx:
                    outputs = model(imgs_dev)

                batch_rows = []
                for (img, target), output in zip(samples, outputs):
                    sample_id = int(target["image_id"].item())
                    original = gt.imgs[sample_id]
                    scaled_h, scaled_w = img.shape[-2:]
                    img_preds = []
                    for box, score, label in zip(output["boxes"].detach().cpu().tolist(),
                                                 output["scores"].detach().cpu().tolist(),
                                                 output["labels"].detach().cpu().tolist()):
                        if not 1 <= int(label) < len(BDD_CLASSES):
                            continue
                        x1, y1, x2, y2 = box
                        sx = float(original["width"]) / scaled_w
                        sy = float(original["height"]) / scaled_h
                        x1, x2 = max(0.0, x1 * sx), min(float(original["width"]), x2 * sx)
                        y1, y2 = max(0.0, y1 * sy), min(float(original["height"]), y2 * sy)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        img_preds.append({
                            "image_id": sample_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        })
                    completed_by_id[sample_id] = img_preds
                    batch_rows.append({
                        "eval_key": expected_key,
                        "image_id": sample_id,
                        "preds": img_preds,
                    })
                    processed_this_call += 1
                append_jsonl_batch(partial_path, batch_rows)
    finally:
        if eval_pool is not None:
            eval_pool.shutdown(wait=True)

    predictions = []
    for sample_id in id_set:
        predictions.extend(completed_by_id.get(sample_id, []))

    atomic_write_json(detections_path, predictions, indent=None)
    if not predictions:
        result = {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
            "images": len(id_set),
            "seconds": round(time.monotonic() - started, 2),
            "notice": "no predicted detections",
            "complete": True,
            "eval_key": expected_key,
        }
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(predictions)
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.imgIds = id_set
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        result = {
            "AP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
            "AP_small": float(evaluator.stats[3]),
            "AP_medium": float(evaluator.stats[4]),
            "AP_large": float(evaluator.stats[5]),
            "images": len(id_set),
            "seconds": round(time.monotonic() - started, 2),
            "complete": True,
            "eval_key": expected_key,
        }
    atomic_write_json(metrics_path, result)
    with contextlib.suppress(OSError):
        if partial_path.is_file():
            partial_path.unlink()
    return result


def build_scientific_identity(args, *, init_checkpoint_sha256: str | None = None) -> dict:
    """Path-independent scientific identity so checkpoints transfer across GPUs/hosts.

    Uses ``effective_batch_size = batch_size * grad_accum_steps`` so a run started
    on Kaggle T4 (2x4) can resume identically on RTX 5090 / L40s (4x2 or 8x1).
    """
    config = vars(args).copy()
    identity = {k: v for k, v in config.items() if k not in _NON_IDENTITY_KEYS}
    identity["effective_batch_size"] = int(getattr(args, "batch_size", 1)) * int(getattr(args, "grad_accum_steps", 1))
    if not identity.get("reserve_calibration"):
        identity.pop("reserve_calibration", None)  # Compatible with older unreserved runs.
    if not identity.get("stratify_calibration"):
        identity.pop("stratify_calibration", None)
    if identity.get("anchor_profile", "default") == "default":
        identity.pop("anchor_profile", None)
    if int(identity.get("zero_box_filter", 1)) == 1:
        identity.pop("zero_box_filter", None)
    if init_checkpoint_sha256:
        identity["init_checkpoint_sha256"] = init_checkpoint_sha256
    return identity


def _load_legacy_last(checkpoint: Path, fingerprint: str):
    """Fallback loader for pre-v2 ``last.pth`` files, verified safely."""
    if not checkpoint.is_file():
        return None, None, []
    sidecar = read_json(checkpoint.with_suffix(".json"))
    if sidecar is not None:
        if sidecar.get("fingerprint") != fingerprint:
            raise IncompatibleCheckpoint(
                f"{checkpoint} fingerprint mismatch ({str(sidecar.get('fingerprint'))[:12]} != {fingerprint[:12]})")
        problem = verify_file(checkpoint, sidecar, deep=True)
        if problem:
            quarantine(checkpoint, problem)
            return None, None, [f"last.pth: {problem} -> quarantined"]
    try:
        state = safe_torch_load(checkpoint)
    except Exception as exc:  # noqa: BLE001
        quarantine(checkpoint, f"last.pth unreadable: {exc!r}")
        return None, None, [f"last.pth: unreadable ({type(exc).__name__}) -> quarantined"]
    if state.get("fingerprint") != fingerprint:
        raise IncompatibleCheckpoint(
            f"{checkpoint} settings or data changed (fingerprint mismatch)")
    info = {"step": int(state["step"]), "path": str(checkpoint), **(sidecar or {})}
    return state, info, ["loaded verified last.pth"]


def _mirror_last_checkpoint(saved_ckpt: Path, out: Path) -> None:
    """Keep out/last.pth (+ last.json) atomically synced without duplicating disk space."""
    from ssod_research.recovery import replace_file
    last_pth = out / "last.pth"
    tmp_link = out / f".last.{os.getpid()}.tmp"
    with contextlib.suppress(OSError):
        if tmp_link.exists():
            tmp_link.unlink()
    try:
        os.link(saved_ckpt, tmp_link)
        replace_file(tmp_link, last_pth)
    except OSError:
        with contextlib.suppress(OSError):
            if tmp_link.exists():
                tmp_link.unlink()
        atomic_copy(saved_ckpt, last_pth)
    sidecar = read_json(saved_ckpt.with_suffix(".json"))
    if sidecar:
        mirror_side = dict(sidecar)
        mirror_side["file"] = "last.pth"
        mirror_side["source_checkpoint"] = saved_ckpt.name
        atomic_write_json(out / "last.json", mirror_side)


def _registry_client(args):
    if not getattr(args, "registry_db", None) or not getattr(args, "experiment_id", None):
        return None
    from ssod_research.registry import ExperimentRegistry
    return ExperimentRegistry(args.registry_db)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TorchVision road SSOD experiment")
    p.add_argument("--data-root", default="data/bdd100k")
    p.add_argument("--train-images", default=None)
    p.add_argument("--val-images", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=None,
                   help="Warm-start model weights from a completed supervised warmup")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--detector", choices=("fasterrcnn_v1", "fasterrcnn_v2"), default="fasterrcnn_v1")
    p.add_argument("--anchor-profile", choices=("default", "road_small"), default="default",
                   help="RPN anchor design: 'default' (32..512) or 'road_small' (12..384 multi-scale).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument(
        "--lr-schedule",
        default="constant",
        choices=("constant", "multistep", "cosine", "warmup_constant", "warmup_cosine", "warmup_multistep"),
    )
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--reserve-calibration", action="store_true",
                   help="Exclude the CoRe calibration fold from supervised sampling, including warm-up.")
    p.add_argument("--stratify-calibration", action="store_true",
                   help="Stratify calibration fold by weather/time-of-day and rare-class presence.")
    p.add_argument("--zero-box-filter", type=int, choices=(0, 1), default=1,
                   help="1 = exclude 0-pseudo-box unlabeled images from hard RPN/ROI loss; 0 = include as negatives.")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--pseudo-threshold", type=float, default=0.7)
    p.add_argument("--min-quality", type=float, default=0.3)
    p.add_argument("--agreement-iou", type=float, default=0.5)
    p.add_argument("--localization-scale", type=float, default=4.0)
    p.add_argument("--localization-enabled", type=int, choices=(0, 1), default=1)
    p.add_argument("--adaptive-strength", type=float, default=0.12)
    p.add_argument("--condition-offset", type=float, default=0.05)
    p.add_argument("--unsup-weight", type=float, default=1.0)
    p.add_argument("--unsup-warmup-steps", type=int, default=0,
                   help="Linearly ramp unsupervised loss weight from 0 -> unsup_weight over N steps.")
    p.add_argument("--consistency-weight", type=float, default=0.25)
    p.add_argument("--box-weight", type=float, default=1.0)
    p.add_argument("--class-weight", type=float, default=1.0)
    p.add_argument("--confidence-weight", type=float, default=0.25)
    p.add_argument("--care-tri-zone", type=int, choices=(0, 1), default=1)
    p.add_argument("--care-nwd", type=int, choices=(0, 1), default=1)
    p.add_argument("--care-visibility-cal", type=int, choices=(0, 1), default=1)
    p.add_argument("--care-cutout", type=int, choices=(0, 1), default=1)
    p.add_argument("--care-giou-weight", type=float, default=0.25)
    p.add_argument("--care-kl-z1-weight", type=float, default=0.15)
    p.add_argument("--care-kl-z2-weight", type=float, default=0.12)
    p.add_argument("--care-temperature", type=float, default=1.5)
    p.add_argument("--conformal-alpha-strict", type=float, default=0.10)
    p.add_argument("--conformal-alpha-loose", type=float, default=0.30)
    p.add_argument("--conformal-every", type=int, default=250)
    p.add_argument("--conformal-cal-fraction", type=float, default=0.08)
    p.add_argument("--conformal-max-cal", type=int, default=256)
    p.add_argument("--conformal-vis-weight", type=int, choices=(0, 1), default=1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--eval-max-images", type=int, default=0,
                   help="0 = entire validation set. Restricted subsets are diagnostics, NOT final AP.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", default="auto", choices=("auto", "off", "fp16", "bf16"))
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--registry-db", default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--stop-after-steps", type=int, default=0,
                   help="Testing/budget helper: gracefully checkpoint and stop after N steps in this invocation.")
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1 or args.max_side < 64:
        p.error("steps, batch size, grad_accum_steps and max_side out of range")
    if args.save_every < 1 or args.eval_every < 1 or args.conformal_every < 1:
        p.error("save_every, eval_every and conformal_every must be >= 1")
    for name in ("pseudo_threshold", "min_quality", "agreement_iou", "ema",
                 "conformal_alpha_strict", "conformal_alpha_loose", "conformal_cal_fraction"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            p.error(f"{name} must be in [0,1]")
    for name in ("lr", "unsup_weight", "unsup_warmup_steps", "consistency_weight", "localization_scale",
                 "adaptive_strength", "box_weight", "class_weight", "confidence_weight"):
        if getattr(args, name) < 0:
            p.error(f"{name} must be nonnegative")
    if args.condition_offset < -0.5 or args.condition_offset > 0.5:
        p.error("condition_offset must be between -0.5 and 0.5")
    return args


def train(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = FileLock(out / ".experiment.lock", purpose=f"train:{args.variant}:seed={args.seed}")
    lock.acquire()
    try:
        return _train_locked(args, out)
    finally:
        lock.release()


def _train_locked(args, out: Path):
    ckpt_mgr = CheckpointManager(out / "checkpoints", keep=max(2, int(getattr(args, "keep_checkpoints", 2))))
    ckpt_mgr.cleanup_temporary()

    previous_metadata = read_json(out / "manifest.json", {}) or {}
    if not args.resume and (ckpt_mgr.latest_info() is not None or (out / "last.pth").exists()):
        raise RuntimeError(
            "Existing checkpoints found. Use --resume or choose a new --output instead of overwriting measurements.")

    seed_all(args.seed, deterministic=bool(getattr(args, "deterministic", False)))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    if device.type == "cpu" and args.steps > 3:
        print("WARNING: CPU FasterRCNN training is slow; use GPU for full experiments", flush=True)

    amp_mode, autocast_enabled, dtype = resolve_amp(args.amp, device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None

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
    if unlabeled is not None:
        named.update(im["file_name"] for im in unlabeled.images)
    if named.intersection(im["file_name"] for im in validation.images):
        raise ValueError("train/validation image leakage")

    attrs = training_attributes(root)
    if args.variant in ("condition", "full") and not attrs:
        raise ValueError("condition variant requires train_attributes.json (re-run preparation)")
    counts = class_frequencies(root)

    dataset_hashes = {p.name: _sha256_file(p) for p in (label_path, val_path)}
    if unlabeled is not None:
        dataset_hashes[unlabeled_path.name] = _sha256_file(unlabeled_path)
    if args.variant in ("condition", "full"):
        dataset_hashes["train_attributes.json"] = _sha256_file(root / "annotations" / "train_attributes.json")
    dataset_fingerprint = stable_hash(dataset_hashes)

    init_sha = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"warm-start checkpoint not found: {init_path}")
        init_sha = _sha256_file(init_path)

    scientific_identity = build_scientific_identity(args, init_checkpoint_sha256=init_sha)
    config_fingerprint = stable_hash(scientific_identity)
    fingerprint = stable_hash({"settings": scientific_identity, "datasets": dataset_hashes})

    config = vars(args).copy()
    config["device"] = str(device)
    config["resolved_amp"] = amp_mode

    hardware_profile = {
        "device_type": device.type,
        "amp": amp_mode,
        "gpu_name": str(gpu_name) if gpu_name is not None else None,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "host": str(host_id()),
    }

    model = build_model(
        pretrained_backbone=args.pretrained_backbone and not bool(args.init_checkpoint),
        max_side=args.max_side,
        detector=getattr(args, "detector", "fasterrcnn_v1"),
        anchor_profile=getattr(args, "anchor_profile", "default"),
    ).to(device)
    teacher = copy.deepcopy(model).to(device) if unlabeled is not None else None
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = build_scheduler(
        optimizer,
        schedule=args.lr_schedule,
        total_steps=args.steps,
        warmup_steps=int(getattr(args, "lr_warmup_steps", 0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_mode == "fp16"))

    # Strictly disjoint calibration split carved ONLY from labeled training data (never val.json)
    if args.reserve_calibration or args.variant == "core_ssod":
        cal_strata = None
        if getattr(args, "stratify_calibration", False):
            rare_cats = {2, 5, 6, 7, 8}  # rider, bus, train, motorcycle, bicycle
            cal_strata = []
            for im in labeled.images:
                at = attrs.get(im["file_name"], {})
                tod = str(at.get("timeofday", "unknown")).strip().lower() or "unknown"
                wea = str(at.get("weather", "unknown")).strip().lower() or "unknown"
                im_anns = labeled.annotations.get(int(im["id"]), [])
                has_rare = any(int(a.get("category_id", 0)) in rare_cats for a in im_anns)
                cal_strata.append(f"{tod}|{wea}|r{int(has_rare)}")
        train_labeled_ids, cal_labeled_ids = split_calibration_indices(
            len(labeled),
            seed=args.seed,
            cal_fraction=float(args.conformal_cal_fraction),
            max_cal=int(args.conformal_max_cal),
            strata=cal_strata,
        )
        if not cal_labeled_ids:
            raise ValueError("Calibration needs at least two labeled images for disjoint train and calibration folds")
    else:
        train_labeled_ids = list(range(len(labeled)))
        cal_labeled_ids = []

    fold_hash = stable_hash([labeled.images[i]["file_name"] for i in cal_labeled_ids]) if cal_labeled_ids else None

    running_stats = {"total_candidates": 0, "total_accepted": 0, "total_quality_sum": 0.0}
    care_calibrator = (
        OnlineVisibilityClassCalibrator(
            num_foreground_classes=len(BDD_CLASSES) - 1,
            base_threshold=args.pseudo_threshold,
        )
        if args.variant == "care_ssod"
        else None
    )
    conformal_calibrator = (
        WeightedConformalRiskCalibrator(
            num_foreground_classes=len(BDD_CLASSES) - 1,
            alpha_strict=float(args.conformal_alpha_strict),
            alpha_loose=float(args.conformal_alpha_loose),
            use_visibility_weighting=bool(args.conformal_vis_weight),
            class_counts=counts,
        )
        if args.variant == "core_ssod"
        else None
    )
    gmm_threshold = (
        DynamicGMMThreshold(
            num_foreground_classes=len(BDD_CLASSES) - 1,
            base_threshold=float(args.pseudo_threshold),
        )
        if args.variant == "consistent_teacher"
        else None
    )
    start = 0
    resume_mode = "fresh"
    resume_warnings: list[str] = []
    recovery_notes: list[str] = []
    loaded_ckpt_info = None
    previous_seconds = float(previous_metadata.get("elapsed_seconds") or 0.0)
    previous_peak = float(previous_metadata.get("gpu_peak_reserved_gib") or 0.0)

    if args.resume:
        state, ckpt_info, notes = ckpt_mgr.load_latest_valid(fingerprint, deep=True)
        recovery_notes.extend(notes)
        if state is None and (out / "last.pth").is_file():
            state, ckpt_info, last_notes = _load_legacy_last(out / "last.pth", fingerprint)
            recovery_notes.extend(last_notes)

        if state is not None and ckpt_info is not None:
            loaded_ckpt_info = ckpt_info
            model.load_state_dict(state["student"], strict=True)
            if teacher is not None:
                if state.get("teacher") is not None:
                    teacher.load_state_dict(state["teacher"], strict=True)
                else:
                    teacher.load_state_dict(state["student"], strict=True)
                    resume_warnings.append("teacher weights missing; re-initialized from student")
            if state.get("optimizer") is not None:
                optimizer.load_state_dict(state["optimizer"])
            else:
                resume_warnings.append("optimizer state missing")
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            else:
                resume_warnings.append("lr scheduler state missing")
            if state.get("scaler") is not None:
                if device.type == "cuda" and amp_mode == "fp16":
                    scaler.load_state_dict(state["scaler"])
            else:
                resume_warnings.append("GradScaler state missing")

            rng_issues = restore_rng_state(state.get("rng_state"), device.type)
            resume_warnings.extend(rng_issues)

            if state.get("sampler_state") is None:
                resume_warnings.append("sampler metadata missing")
            alg = state.get("algorithm_state")
            if isinstance(alg, dict):
                if isinstance(alg.get("running_stats"), dict):
                    running_stats.update(alg["running_stats"])
                if care_calibrator is not None:
                    if isinstance(alg.get("care_calibrator"), dict):
                        care_calibrator.load_state_dict(alg["care_calibrator"])
                    else:
                        resume_warnings.append("care_ssod online calibrator state missing")
                if conformal_calibrator is not None:
                    if isinstance(alg.get("conformal_calibrator"), dict):
                        conformal_calibrator.load_state_dict(alg["conformal_calibrator"])
                    else:
                        resume_warnings.append("core_ssod conformal calibrator state missing")
                if gmm_threshold is not None:
                    if isinstance(alg.get("gmm_threshold"), dict):
                        gmm_threshold.load_state_dict(alg["gmm_threshold"])
                    else:
                        resume_warnings.append("consistent_teacher GMM threshold state missing")
            else:
                resume_warnings.append("algorithm mutable state missing")

            saved_hw = state.get("hardware") or {}
            if saved_hw:
                if saved_hw.get("device_type") != device.type or saved_hw.get("amp") != amp_mode:
                    resume_warnings.append(
                        f"hardware/precision changed: {saved_hw.get('device_type')}/{saved_hw.get('amp')} "
                        f"({saved_hw.get('gpu_name')}) -> {device.type}/{amp_mode} ({gpu_name})"
                    )
                elif saved_hw.get("gpu_name") != gpu_name:
                    resume_warnings.append(
                        f"GPU model changed: {saved_hw.get('gpu_name')} -> {gpu_name}"
                    )
            else:
                resume_warnings.append("checkpoint missing hardware profile metadata")

            metrics_state = state.get("metrics_state") or {}
            previous_seconds = max(previous_seconds, float(metrics_state.get("elapsed_seconds") or 0.0))
            previous_peak = max(previous_peak, float(metrics_state.get("gpu_peak_reserved_gib") or 0.0))

            start = int(state["step"])
            if start > args.steps:
                raise RuntimeError(f"Saved checkpoint step {start} exceeds requested step budget {args.steps}")
            resume_mode = "exact" if not resume_warnings else "approximate"
            print(
                f"Resuming ({resume_mode}) from step {start}/{args.steps} using {ckpt_info['path']}",
                flush=True,
            )
            for note in recovery_notes:
                print(f"RECOVERY: {note}", flush=True)
            for warn in resume_warnings:
                print(f"RESUME_NOTICE: {warn}", flush=True)
        elif previous_metadata.get("status") in ("started", "interrupted", "failed", "completed"):
            if previous_metadata.get("fingerprint") and previous_metadata.get("fingerprint") != fingerprint:
                raise IncompatibleCheckpoint(
                    "Existing manifest belongs to a different configuration/dataset fingerprint")
            resume_mode = "fresh_after_missing_checkpoint"
            resume_warnings.append("No valid checkpoint found; restarting from step 0")
            print("WARNING: No valid checkpoint found in run directory; restarting from step 0", flush=True)

    conformal_exchangeability_valid = True if args.variant == "core_ssod" else None
    if start == 0 and args.init_checkpoint:
        warmup = safe_torch_load(args.init_checkpoint)
        if warmup.get("step", 0) < 1 or "student" not in warmup:
            raise ValueError("warm-start checkpoint must be from a successful supervised run")
        warmup_cal_matches = (
            warmup.get("calibration_fold_sha256") == fold_hash
            and bool(warmup.get("config", {}).get("reserve_calibration"))
            and warmup.get("config", {}).get("variant") == "supervised"
            and warmup.get("dataset_hashes", {}).get(label_path.name) == dataset_hashes[label_path.name]
        )
        if args.reserve_calibration and not warmup_cal_matches:
            raise ValueError("Warm-up must reserve the same calibration images from the same labeled split.")
        if args.variant == "core_ssod":
            conformal_exchangeability_valid = bool(warmup_cal_matches)
            if not warmup_cal_matches:
                print(
                    "SCIENTIFIC_WARNING: Warm-start checkpoint did NOT reserve D_cal (--reserve-calibration). "
                    "Conformal calibration fold was seen during supervised burn-in!",
                    flush=True,
                )
        model.load_state_dict(warmup["student"], strict=True)
        if teacher is not None:
            teacher.load_state_dict(warmup["student"], strict=True)
        print(f"Loaded supervised warm-start: {args.init_checkpoint}", flush=True)

    step_log = out / "steps.jsonl"
    eval_log = out / "evaluations.jsonl"
    conformal_log = out / "conformal_calibration.jsonl"
    if start > 0:
        truncate_jsonl(step_log, start)
        truncate_jsonl(eval_log, start)
        truncate_jsonl(conformal_log, start)
    elif resume_mode == "fresh_after_missing_checkpoint":
        truncate_jsonl(step_log, 0)
        truncate_jsonl(eval_log, 0)
        truncate_jsonl(conformal_log, 0)

    metadata = {
        "created_utc": previous_metadata.get("created_utc") or utc_now(),
        "updated_utc": utc_now(),
        "arguments": config,
        "source_revision": source_revision(),
        "calibration_fold_sha256": fold_hash,
        "conformal_exchangeability_valid": conformal_exchangeability_valid,
        "scientific_identity": scientific_identity,
        "fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "data": inventories,
        "supervised_pool_images": len(train_labeled_ids),
        "calibration_pool_images": len(cal_labeled_ids),
        "dataset_sha256": dataset_hashes,
        "hardware": hardware_profile,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "status": "running",
        "resume_mode": resume_mode,
        "resume_warnings": resume_warnings,
        "recovery_notes": recovery_notes,
        "last_completed_step": start,
        "target_steps": int(args.steps),
        "last_checkpoint": loaded_ckpt_info.get("path") if loaded_ckpt_info else None,
        "last_checkpoint_sha256": loaded_ckpt_info.get("sha256") if loaded_ckpt_info else None,
        "elapsed_seconds": round(previous_seconds, 2),
        "gpu_peak_reserved_gib": round(previous_peak, 3),
        "framework": "TorchVision native research implementation; not SoftTeacher reproduction",
    }
    atomic_write_json(out / "manifest.json", metadata)

    registry = _registry_client(args)
    if registry is not None:
        with contextlib.suppress(KeyError):
            registry.mark_running(
                args.experiment_id,
                gpu_name=gpu_name,
                config_fingerprint=config_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                resume_mode=resume_mode,
            )

    started = time.monotonic()
    last_saved_step = start
    current_step = start
    stop_signal: list[int] = []

    def _on_signal(signum, _frame):
        if stop_signal:
            raise KeyboardInterrupt(f"Forced exit on repeated signal {signum}")
        stop_signal.append(signum)
        print(f"\nSignal {signum} received; saving checkpoint cleanly after current step...", flush=True)

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    try:
        signal.signal(signal.SIGINT, _on_signal)
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        prev_sigint = None
        prev_sigterm = None

    grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    eff_batch = int(args.batch_size) * grad_accum

    def _save_full_checkpoint(step_num: int):
        nonlocal last_saved_step
        elapsed_total = round(previous_seconds + (time.monotonic() - started), 2)
        peak_gib = round(
            max(previous_peak, (torch.cuda.max_memory_reserved() / 2**30) if device.type == "cuda" else 0.0),
            3,
        )
        epoch_float = round((step_num * eff_batch) / max(1, len(train_labeled_ids)), 6)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "config_fingerprint": config_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_hashes": dataset_hashes,
            "config": scientific_identity,
            "step": int(step_num),
            "epoch": epoch_float,
            "student": model.state_dict(),
            "source_revision": metadata["source_revision"],
            "calibration_fold_sha256": fold_hash,
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "sampler_state": {
                "type": "stateless_step_rng_v1",
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "labeled_size": len(labeled),
                "supervised_pool_size": len(train_labeled_ids),
                "unlabeled_size": len(unlabeled) if unlabeled is not None else 0,
                "next_step": int(step_num),
            },
            "algorithm_state": {
                "variant": args.variant,
                "ema_decay": float(args.ema),
                "pseudo_threshold": float(args.pseudo_threshold),
                "adaptive_strength": float(args.adaptive_strength),
                "condition_offset": float(args.condition_offset),
                "class_counts": counts,
                "running_stats": dict(running_stats),
                "care_calibrator": care_calibrator.state_dict() if care_calibrator is not None else None,
                "conformal_calibrator": conformal_calibrator.state_dict() if conformal_calibrator is not None else None,
                "gmm_threshold": gmm_threshold.state_dict() if gmm_threshold is not None else None,
            },
            "metrics_state": {
                "elapsed_seconds": elapsed_total,
                "gpu_peak_reserved_gib": peak_gib,
            },
            "hardware": hardware_profile,
        }
        saved_path = ckpt_mgr.save(
            payload,
            step_num,
            sidecar={
                "fingerprint": fingerprint,
                "config_fingerprint": config_fingerprint,
                "dataset_fingerprint": dataset_fingerprint,
                "variant": args.variant,
                "seed": int(args.seed),
                "hardware": hardware_profile,
            },
        )
        _mirror_last_checkpoint(saved_path, out)
        sidecar = read_json(saved_path.with_suffix(".json"), {}) or {}
        last_saved_step = step_num
        metadata["last_completed_step"] = step_num
        metadata["last_checkpoint"] = str(saved_path)
        metadata["last_checkpoint_sha256"] = sidecar.get("sha256")
        metadata["elapsed_seconds"] = elapsed_total
        metadata["gpu_peak_reserved_gib"] = peak_gib
        metadata["updated_utc"] = utc_now()
        atomic_write_json(out / "manifest.json", metadata)
        if registry is not None:
            with contextlib.suppress(KeyError):
                registry.update_progress(
                    args.experiment_id,
                    last_completed_step=step_num,
                    checkpoint_path=str(saved_path),
                    checkpoint_sha256=sidecar.get("sha256"),
                    resume_mode=resume_mode,
                    accumulated_runtime_sec=elapsed_total,
                    gpu_name=gpu_name,
                )
        return saved_path

    def _run_evaluation(step_num: int):
        # Preserve training RNG across evaluation so eval_every never breaks exact determinism.
        saved_rng = capture_rng_state()
        try:
            evaluated = teacher if teacher is not None else model
            eval_fp = f"{fingerprint}:{step_num}:{args.eval_max_images}"
            metrics = evaluate(
                evaluated,
                validation,
                device,
                out / f"eval_{step_num:07d}",
                max_images=args.eval_max_images or None,
                eval_fingerprint=eval_fp,
                resume_partial=True,
                amp_dtype=(dtype if autocast_enabled else None),
            )
        finally:
            restore_rng_state(saved_rng, device.type)
        metrics_row = dict(metrics)
        metrics_row.update({"step": int(step_num), "partial_validation": args.eval_max_images > 0})
        existing_evals = read_jsonl(eval_log)
        if not any(int(r.get("step", -1)) == int(step_num) for r in existing_evals):
            append_jsonl(eval_log, metrics_row)
        print("EVAL", json.dumps(metrics_row), flush=True)
        return metrics_row

    def _run_conformal_calibration(step_num: int):
        if conformal_calibrator is None or teacher is None or not cal_labeled_ids:
            return None
        saved_rng = capture_rng_state()
        try:
            diag = conformal_calibrator.calibrate_on_dataset(
                teacher,
                labeled,
                cal_labeled_ids,
                device,
                step=step_num,
                max_samples=int(args.conformal_max_cal),
                batch_size=max(1, min(8, eff_batch)),
                amp_dtype=(dtype if autocast_enabled else None),
                use_nwd=bool(args.care_nwd),
            )
        finally:
            restore_rng_state(saved_rng, device.type)
        append_jsonl(conformal_log, diag)
        print("CONFORMAL", json.dumps(diag), flush=True)
        return diag

    if conformal_calibrator is not None and conformal_calibrator.calibration_count == 0 and start < args.steps:
        _run_conformal_calibration(start)

    io_pool = (
        ThreadPoolExecutor(max_workers=min(6, max(2, os.cpu_count() or 4)))
        if device.type == "cuda"
        else None
    )

    def _prefetch_step(step_idx: int):
        sup_slots = sample_indices(len(train_labeled_ids), eff_batch, args.seed, step_idx, 0)
        sup_ids = [train_labeled_ids[idx] for idx in sup_slots]
        u_ids = (
            sample_indices(len(unlabeled), eff_batch, args.seed, step_idx, 1)
            if teacher is not None else []
        )
        if io_pool is not None:
            sup_futs = [io_pool.submit(labeled.__getitem__, idx) for idx in sup_ids]
            u_futs = [io_pool.submit(unlabeled.__getitem__, idx) for idx in u_ids] if u_ids else []
            return sup_ids, u_ids, sup_futs, u_futs
        return sup_ids, u_ids, None, None

    steps_this_call = 0
    prefetched = _prefetch_step(start) if start < args.steps else None
    try:
        for step in range(start, args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            assert prefetched is not None
            sup_all_ids, u_all_ids, sup_futs, u_futs = prefetched
            if sup_futs is not None:
                sup_step_data = [f.result() for f in sup_futs]
                u_step_data = [f.result() for f in u_futs] if u_futs else []
            else:
                sup_step_data = cpu_batch(labeled, sup_all_ids)
                u_step_data = cpu_batch(unlabeled, u_all_ids) if u_all_ids else []

            # Overlap PIL decode/resize of step+1 with GPU forward+backward of current step
            prefetched = _prefetch_step(step + 1) if (step + 1) < args.steps else None

            unsup_ramp = (
                min(1.0, float(step + 1) / float(args.unsup_warmup_steps))
                if int(args.unsup_warmup_steps) > 0
                else 1.0
            )
            eff_unsup_weight = float(args.unsup_weight) * unsup_ramp

            step_loss_sum = 0.0
            sup_sum = 0.0
            unsup_sum = 0.0
            stab_sum = 0.0
            extra: dict = {
                "sup": 0.0, "unsup": 0.0, "stability": 0.0,
                "candidates": 0, "accepted": 0,
            }
            if teacher is not None and int(args.unsup_warmup_steps) > 0:
                extra["unsup_w"] = round(eff_unsup_weight, 4)

            for micro in range(grad_accum):
                sl = slice(micro * args.batch_size, (micro + 1) * args.batch_size)
                raw_train_batch = sup_step_data[sl]
                train_batch = []
                for b_idx, (img_cpu, tgt_cpu) in enumerate(raw_train_batch):
                    slot_idx = micro * args.batch_size + b_idx
                    rng_sup = random.Random(args.seed * 5381 + step * 97 + slot_idx * 13 + 1)
                    train_batch.append(_weak_flip_sample(img_cpu, tgt_cpu, rng_sup.random() < 0.5))
                images = [img.to(device) for img, _ in train_batch]
                targets = portable_targets(train_batch, device)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    sup_losses = model(images, targets)
                    supervised = sum(sup_losses.values())
                    micro_total = supervised
                sup_sum += float(supervised.detach()) / grad_accum

                if teacher is not None:
                    ids = u_all_ids[sl]
                    batch_u = u_step_data[sl]
                    originals = [img.to(device) for img, _ in batch_u]
                    rng_micro = random.Random(args.seed * 7103 + step * 61 + micro * 17 + 3)

                    if args.variant in ("care_ssod", "core_ssod"):
                        active_cal = conformal_calibrator if args.variant == "core_ssod" else care_calibrator
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            tri_outputs, statistics, t_cache = extract_teacher_tri_zone_targets(
                                teacher,
                                originals,
                                active_cal,
                                use_nwd=bool(args.care_nwd),
                                use_visibility_calibration=(
                                    bool(args.conformal_vis_weight)
                                    if args.variant == "core_ssod"
                                    else bool(args.care_visibility_cal)
                                ),
                                use_tri_zone=bool(args.care_tri_zone),
                                return_cache=True,
                            )
                        extra["candidates"] += int(statistics.get("candidates", 0))
                        extra["accepted"] += int(statistics.get("accepted", 0))
                        extra["zone1_count"] = int(extra.get("zone1_count", 0)) + int(statistics.get("zone1_count", 0))
                        extra["zone2_count"] = int(extra.get("zone2_count", 0)) + int(statistics.get("zone2_count", 0))
                        extra["degradation_mean"] = statistics.get("degradation_mean", 0.0)
                        running_stats["total_candidates"] += int(statistics.get("candidates", 0))
                        running_stats["total_accepted"] += int(statistics.get("accepted", 0))
                        running_stats["total_quality_sum"] += float(statistics.get("quality_sum", 0.0))

                        aug_images, z1_targets, z1_meta, z2_meta, flips = [], [], [], [], []
                        for b_idx, (img, (_, target), item) in enumerate(zip(originals, batch_u, tri_outputs)):
                            slot_idx = micro * args.batch_size + b_idx
                            img_rng = random.Random(args.seed * 7103 + step * 61 + slot_idx * 17 + 3)
                            a_img, a_z1, a_z2, was_flipped = strong_view_care(
                                img, item["zone1"], item["zone2"],
                                rng=img_rng, enable_cutout=bool(args.care_cutout),
                                return_flip=True,
                            )
                            aug_images.append(a_img)
                            z1_targets.append(target_for_training(a_z1, image_id=int(target["image_id"])))
                            z1_meta.append(a_z1)
                            z2_meta.append(a_z2)
                            flips.append(was_flipped)

                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            care_loss, care_extra = care_unsupervised_loss(
                                model,
                                aug_images,
                                z1_targets,
                                z1_meta,
                                z2_meta,
                                teacher=teacher,
                                teacher_cache=t_cache,
                                flips=flips,
                                class_counts=counts,
                                use_tri_zone=bool(args.care_tri_zone),
                                giou_weight=float(args.care_giou_weight),
                                kl_zone1_weight=float(args.care_kl_z1_weight),
                                kl_zone2_weight=float(args.care_kl_z2_weight),
                                temperature=float(args.care_temperature),
                                filter_zero_box_images=bool(getattr(args, "zero_box_filter", 1)),
                            )
                            if care_loss is not None and eff_unsup_weight > 0.0:
                                micro_total = micro_total + eff_unsup_weight * care_loss
                            unsup_sum += float(care_extra.get("unsup", 0.0)) / grad_accum
                            stab_sum += float(care_extra.get("stability", 0.0)) / grad_accum
                            extra["suppressed_bg"] = int(extra.get("suppressed_bg", 0)) + int(care_extra.get("suppressed_bg", 0))

                    elif args.variant == "soft_teacher":
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            st_pseudo, statistics, t_cache = soft_teacher_extract(
                                teacher,
                                originals,
                                cls_threshold=float(args.pseudo_threshold),
                                rng=rng_micro,
                            )
                        extra["candidates"] += int(statistics.get("candidates", 0))
                        extra["accepted"] += int(statistics.get("accepted", 0))
                        running_stats["total_candidates"] += int(statistics.get("candidates", 0))
                        running_stats["total_accepted"] += int(statistics.get("accepted", 0))
                        running_stats["total_quality_sum"] += float(statistics.get("quality_sum", 0.0))

                        aug_images, aug_targets, flips, cls_valid_list, reg_valid_list = [], [], [], [], []
                        for b_idx, (img, (_, target), p) in enumerate(zip(originals, batch_u, st_pseudo)):
                            slot_idx = micro * args.batch_size + b_idx
                            img_rng = random.Random(args.seed * 7103 + step * 61 + slot_idx * 17 + 5)
                            flip = img_rng.random() < 0.5
                            brightness = img_rng.uniform(0.75, 1.25)
                            contrast = img_rng.uniform(0.75, 1.25)
                            pt = target_for_training(p, image_id=int(target["image_id"]))
                            a_img, trans = strong_view(img, pt, flip=flip, brightness=brightness, contrast=contrast)
                            aug_images.append(a_img)
                            aug_targets.append(target_for_training(trans, image_id=int(target["image_id"])))
                            flips.append(flip)
                            cls_valid_list.append(p["cls_valid"])
                            reg_valid_list.append(p["reg_valid"])

                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            st_loss, st_extra = soft_teacher_unsupervised_loss(
                                model,
                                teacher,
                                aug_images,
                                aug_targets,
                                flips,
                                reg_valid_list,
                                t_cache,
                                cls_valid_list=cls_valid_list,
                                filter_zero_box_images=bool(getattr(args, "zero_box_filter", 1)),
                            )
                            if st_loss is not None and eff_unsup_weight > 0.0:
                                micro_total = micro_total + eff_unsup_weight * st_loss
                            unsup_sum += float(st_extra.get("unsup", 0.0)) / grad_accum

                    elif args.variant == "consistent_teacher":
                        teacher.eval()
                        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            raw_preds = teacher(originals)
                        aug_images, aug_targets = [], []
                        for b_idx, (img, (_, target), pred) in enumerate(zip(originals, batch_u, raw_preds)):
                            slot_idx = micro * args.batch_size + b_idx
                            img_rng = random.Random(args.seed * 7103 + step * 61 + slot_idx * 17 + 7)
                            scores = pred["scores"]
                            labels = pred["labels"]
                            boxes = pred["boxes"]
                            s_list = scores.detach().cpu().tolist()
                            l_list = labels.detach().cpu().tolist()
                            gmm_threshold.update_and_fit(l_list, s_list)
                            keep = [
                                j for j, (cls_id, sc) in enumerate(zip(l_list, s_list))
                                if math.isfinite(sc) and sc >= gmm_threshold.threshold_for(cls_id)
                            ]
                            extra["candidates"] += len(s_list)
                            extra["accepted"] += len(keep)
                            running_stats["total_candidates"] += len(s_list)
                            running_stats["total_accepted"] += len(keep)
                            if keep:
                                sel = torch.tensor(keep, dtype=torch.long, device=boxes.device)
                                z1 = {"boxes": boxes[sel].detach(), "labels": labels[sel].detach(), "scores": scores[sel].detach()}
                                running_stats["total_quality_sum"] += float(scores[sel].sum().item())
                            else:
                                z1 = {
                                    "boxes": boxes.new_zeros((0, 4)),
                                    "labels": labels.new_zeros((0,), dtype=torch.long),
                                    "scores": scores.new_zeros((0,)),
                                }
                            empty_z2 = {"boxes": boxes.new_zeros((0, 4))}
                            a_img, a_z1, _ = strong_view_care(img, z1, empty_z2, rng=img_rng, enable_cutout=True)
                            if bool(getattr(args, "zero_box_filter", 1)) and not len(a_z1["boxes"]):
                                continue
                            aug_images.append(a_img)
                            aug_targets.append(target_for_training(a_z1, image_id=int(target["image_id"])))
                        if any(len(t["boxes"]) > 0 for t in aug_targets):
                            act_scale = float(len(aug_images)) / max(1.0, float(len(originals)))
                            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                                unsup_losses = model(aug_images, aug_targets)
                                unsupervised = act_scale * sum(unsup_losses.values())
                                if eff_unsup_weight > 0.0:
                                    micro_total = micro_total + eff_unsup_weight * unsupervised
                                unsup_sum += float(unsupervised.detach()) / grad_accum

                    else:
                        attributes = [attrs.get(unlabeled.images[idx]["file_name"], {}) for idx in ids]
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                            pseudo, statistics = pseudo_targets(
                                teacher, originals, variant=args.variant, threshold=args.pseudo_threshold,
                                min_quality=args.min_quality, localization_scale=args.localization_scale,
                                agreement_iou=args.agreement_iou, counts=counts,
                                localization_enabled=bool(args.localization_enabled),
                                adaptive_strength=args.adaptive_strength, attrs_list=attributes,
                                condition_offset=args.condition_offset)
                        extra["candidates"] += int(statistics.get("candidates", 0))
                        extra["accepted"] += int(statistics.get("accepted", 0))
                        extra["quality_sum"] = float(extra.get("quality_sum", 0.0)) + float(statistics.get("quality_sum", 0.0))
                        running_stats["total_candidates"] += int(statistics.get("candidates", 0))
                        running_stats["total_accepted"] += int(statistics.get("accepted", 0))
                        running_stats["total_quality_sum"] += float(statistics.get("quality_sum", 0.0))

                        augmented = []
                        for b_idx, (img, (_, target), p) in enumerate(zip(originals, batch_u, pseudo)):
                            slot_idx = micro * args.batch_size + b_idx
                            img_rng = random.Random(args.seed * 7103 + step * 61 + slot_idx * 17 + 9)
                            flip = img_rng.random() < 0.5
                            brightness = img_rng.uniform(0.75, 1.25)
                            contrast = img_rng.uniform(0.75, 1.25)
                            pseudo_target = target_for_training(p, image_id=int(target["image_id"]))
                            img, trans = strong_view(img, pseudo_target, flip=flip,
                                                     brightness=brightness, contrast=contrast)
                            trans["scores"] = p["scores"]
                            trans["uncertainty"] = p["uncertainty"]
                            if not bool(getattr(args, "zero_box_filter", 1)) or len(trans["boxes"]):
                                augmented.append((img, trans))
                        if any(len(t["boxes"]) > 0 for _, t in augmented):
                            act_scale = float(len(augmented)) / max(1.0, float(len(originals)))
                            aug_images = [item[0] for item in augmented]
                            aug_targets = [target_for_training(item[1], image_id=int(item[1]["image_id"]))
                                           for item in augmented]
                            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                                unsup_losses = model(aug_images, aug_targets)
                                unsupervised = act_scale * sum(unsup_losses.values())
                                if eff_unsup_weight > 0.0:
                                    micro_total = micro_total + eff_unsup_weight * unsupervised
                                unsup_sum += float(unsupervised.detach()) / grad_accum
                                if args.variant == "full" and args.consistency_weight:
                                    stability = consistency_roi_loss(
                                        model, aug_images, [t for _, t in augmented],
                                        box_weight=args.box_weight, class_weight=args.class_weight,
                                        confidence_weight=args.confidence_weight,
                                        reliability_scale=args.localization_scale)
                                    if stability:
                                        stab_total = act_scale * sum(stability.values())
                                        micro_total = micro_total + args.consistency_weight * unsup_ramp * stab_total
                                        stab_sum += float(stab_total.detach()) / grad_accum

                if not torch.isfinite(micro_total):
                    raise RuntimeError(f"non-finite loss at step {step + 1} (micro {micro + 1}/{grad_accum})")
                step_loss_sum += float(micro_total.detach()) / grad_accum
                scaler.scale(micro_total / float(grad_accum)).backward()

            extra["sup"] = sup_sum
            extra["unsup"] = unsup_sum
            extra["stability"] = stab_sum

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
                scheduler.step()
                if teacher is not None:
                    ema_update(teacher, model, min(args.ema, 1 - 1 / (step + 2)))

            current_step = step + 1
            steps_this_call += 1
            if conformal_calibrator is not None and current_step % args.conformal_every == 0 and current_step < args.steps:
                _run_conformal_calibration(current_step)

            entry = {
                "step": current_step,
                "loss": round(step_loss_sum, 7),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(time.monotonic() - started, 2),
                **extra,
            }
            append_jsonl(step_log, entry)
            if step == start or current_step % max(1, min(args.save_every, 25)) == 0:
                print(json.dumps(entry), flush=True)

            must_save = (current_step % args.save_every == 0) or (current_step == args.steps)
            budget_hit = args.stop_after_steps > 0 and steps_this_call >= args.stop_after_steps
            if must_save or stop_signal or budget_hit:
                _save_full_checkpoint(current_step)

            if stop_signal or (budget_hit and current_step < args.steps):
                reason = f"signal {stop_signal[0]}" if stop_signal else f"stop_after_steps={args.stop_after_steps}"
                raise KeyboardInterrupt(f"Graceful checkpointed stop at step {current_step} ({reason})")

            if current_step % args.eval_every == 0 or current_step == args.steps:
                _run_evaluation(current_step)

        # Ensure final evaluation exists even if training finished in an earlier
        # run and crashed during final evaluation before completion was recorded.
        final_eval_done = any(int(r.get("step", -1)) == int(args.steps) for r in read_jsonl(eval_log))
        final_dir = out / f"eval_{args.steps:07d}"
        if (
            not final_eval_done
            or not (read_json(final_dir / "metrics.json") or {}).get("complete")
            or not (final_dir / "detections.json").is_file()
        ):
            _run_evaluation(args.steps)

        metadata["status"] = "completed"
        metadata["last_completed_step"] = int(args.steps)
        if registry is not None:
            with contextlib.suppress(KeyError):
                registry.mark_completed(
                    args.experiment_id,
                    last_completed_step=int(args.steps),
                    checkpoint_path=metadata.get("last_checkpoint"),
                    checkpoint_sha256=metadata.get("last_checkpoint_sha256"),
                    artifacts={
                        "manifest": str(out / "manifest.json"),
                        "evaluations": str(eval_log),
                        "final_detections": str(out / f"eval_{args.steps:07d}" / "detections.json"),
                        "last_checkpoint": str(out / "last.pth"),
                    },
                    accumulated_runtime_sec=round(previous_seconds + (time.monotonic() - started), 2),
                    config_fingerprint=config_fingerprint,
                    dataset_fingerprint=dataset_fingerprint,
                    resume_mode=resume_mode,
                )
        return metadata
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metadata["status"] = "failed"
        metadata["error_type"] = "CUDA_OOM"
        metadata["exception"] = repr(exc)
        if registry is not None:
            with contextlib.suppress(KeyError):
                registry.mark_failed(
                    args.experiment_id,
                    f"CUDA_OOM at step {current_step}: {exc}",
                    status="FAILED",
                    last_completed_step=last_saved_step,
                    checkpoint_path=metadata.get("last_checkpoint"),
                    checkpoint_sha256=metadata.get("last_checkpoint_sha256"),
                    accumulated_runtime_sec=round(previous_seconds + (time.monotonic() - started), 2),
                )
        raise
    except KeyboardInterrupt as exc:
        if current_step > last_saved_step:
            with contextlib.suppress(Exception):
                _save_full_checkpoint(current_step)
        metadata["status"] = "interrupted"
        metadata["exception"] = repr(exc)
        if registry is not None:
            with contextlib.suppress(KeyError):
                registry.mark_failed(
                    args.experiment_id,
                    str(exc) or "Interrupted",
                    status="INTERRUPTED",
                    last_completed_step=last_saved_step,
                    checkpoint_path=metadata.get("last_checkpoint"),
                    checkpoint_sha256=metadata.get("last_checkpoint_sha256"),
                    accumulated_runtime_sec=round(previous_seconds + (time.monotonic() - started), 2),
                )
        raise
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["exception"] = repr(exc)
        if registry is not None:
            with contextlib.suppress(KeyError):
                registry.mark_failed(
                    args.experiment_id,
                    repr(exc),
                    status="FAILED",
                    last_completed_step=last_saved_step,
                    checkpoint_path=metadata.get("last_checkpoint"),
                    checkpoint_sha256=metadata.get("last_checkpoint_sha256"),
                    accumulated_runtime_sec=round(previous_seconds + (time.monotonic() - started), 2),
                )
        raise
    finally:
        if io_pool is not None:
            io_pool.shutdown(wait=True)
        if prev_sigint is not None:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGINT, prev_sigint)
        if prev_sigterm is not None:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGTERM, prev_sigterm)
        metadata["ended_utc"] = utc_now()
        metadata["updated_utc"] = utc_now()
        metadata["elapsed_seconds"] = round(previous_seconds + (time.monotonic() - started), 2)
        if device.type == "cuda":
            metadata["gpu_peak_reserved_gib"] = round(
                max(previous_peak, torch.cuda.max_memory_reserved() / 2**30), 3)
        atomic_write_json(out / "manifest.json", metadata)


if __name__ == "__main__":
    train(parse_args())
