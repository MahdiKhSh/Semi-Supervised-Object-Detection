"""Geometric utilities with no assumptions about detector architecture.

Homographies map original image coordinates to augmented-view coordinates.
Boxes use xyxy order. Invalid/projectively clipped boxes are filtered.
"""
from __future__ import annotations

import numpy as np


def project_boxes(boxes, homography, image_size=None, eps=1e-8):
    """Project all four corners of each xyxy box, not merely two diagonals.

    Args:
        boxes: (N,4), xyxy.
        homography: (3,3) projective transform source -> target.
        image_size: optional (height,width) target clip limits.
    Returns:
        (projected_boxes, valid_mask). Invalid entries contain zeroes.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    n = len(b)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=bool)
    corners = np.stack([
        b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]
    ], axis=1)
    homog = np.concatenate([corners, np.ones((n, 4, 1))], axis=-1)
    mapped = homog @ h.T
    z = mapped[..., 2]
    valid = np.isfinite(mapped).all(axis=(1, 2)) & (np.abs(z) > eps).all(axis=1)
    xy = np.divide(mapped[..., :2], z[..., None], out=np.zeros((n, 4, 2)), where=np.abs(z[..., None]) > eps)
    out = np.concatenate([xy.min(axis=1), xy.max(axis=1)], axis=1)
    if image_size is not None:
        height, width = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    valid &= np.isfinite(out).all(axis=1) & (out[:, 2] > out[:, 0] + eps) & (out[:, 3] > out[:, 1] + eps)
    valid &= (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    out[~valid] = 0
    return out, valid


def pairwise_iou(boxes_a, boxes_b):
    """Return NxM IoUs, handling empty arrays and degenerate boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(hi - lo, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_class_matches(boxes_a, labels_a, boxes_b, labels_b, minimum_iou=0.5):
    """Unique 1:1 matches (a_idx,b_idx,iou), highest IoU first, class constrained.

    For diagnostics and research prototyping; not a differentiable matching loss.
    """
    iou = pairwise_iou(boxes_a, boxes_b)
    la, lb = np.asarray(labels_a).reshape(-1), np.asarray(labels_b).reshape(-1)
    if iou.shape != (len(la), len(lb)):
        raise ValueError("box/label lengths mismatch")
    matches, taken_a, taken_b = [], set(), set()
    candidates = np.argwhere((iou >= minimum_iou) & (la[:, None] == lb[None, :]))
    for i, j in sorted(candidates.tolist(), key=lambda pair: (-iou[pair[0], pair[1]], pair[0], pair[1])):
        if i not in taken_a and j not in taken_b:
            matches.append((int(i), int(j), float(iou[i, j])))
            taken_a.add(i)
            taken_b.add(j)
    return matches
