"""Experimental SoftTeacher subclass for MMDetection 3.x.

Import this module via custom_imports in config. Do not copy upstream code.
The subclass changes pseudo-label score gating only: the original SoftTeacher
continues to use its existing classification and regression losses.
"""
from __future__ import annotations

import torch
from mmdet.models.detectors.soft_teacher import SoftTeacher
from mmdet.registry import MODELS


@MODELS.register_module()
class ReliabilitySoftTeacher(SoftTeacher):
    @torch.no_grad()
    def get_pseudo_instances(self, batch_inputs, batch_data_samples):
        batch_data_samples, info = super().get_pseudo_instances(batch_inputs, batch_data_samples)
        cfg = self.semi_train_cfg
        if not cfg.get("enable_reliability", True):
            return batch_data_samples, info
        scale = float(cfg.get("reliability_loc_scale", 4.0))
        minimum = float(cfg.get("reliability_min_score", 0.5))
        min_quality = float(cfg.get("reliability_min_quality", 0.3))
        per_class = cfg.get("reliability_class_thresholds", None)
        for sample in batch_data_samples:
            instances = sample.gt_instances
            scores = instances.scores
            if scores.numel() == 0:
                continue
            uncertainty = instances.reg_uncs
            if uncertainty.shape[0] != scores.shape[0]:
                raise RuntimeError("reg_uncs and scores do not align")
            valid = torch.isfinite(scores) & torch.isfinite(uncertainty) & (uncertainty >= 0)
            quality = scores * torch.exp(-scale * torch.clamp(uncertainty, min=0))
            quality = torch.where(valid, quality, torch.zeros_like(quality))
            if per_class is None:
                thresholds = torch.full_like(scores, minimum)
            else:
                table = scores.new_tensor(per_class)
                if instances.labels.min() < 0 or instances.labels.max() >= table.numel():
                    raise ValueError("class labels outside configured thresholds")
                thresholds = table[instances.labels.long()]
            # Gating is performed *before* SoftTeacher's own later filtering.
            # Preserve the true classifier score rather than pretending
            # that the quality score is a calibrated class probability.
            keep = valid & (scores >= thresholds) & (quality >= min_quality)
            sample.gt_instances = instances[keep]
        return batch_data_samples, info
