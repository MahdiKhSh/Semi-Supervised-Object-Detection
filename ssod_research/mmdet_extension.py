"""Research SoftTeacher subclass registered with MMDetection 3.x.

Reuses upstream SoftTeacher rather than vendoring copyrighted framework code.
Adds reliability filtering and an ablatable differentiable three-term
ROI consistency regularizer; mathematical novelty has NOT been established.
"""
from __future__ import annotations

import torch
from mmdet.models.detectors.soft_teacher import SoftTeacher
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from ssod_research.consistency import weighted_consistency_loss


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
            keep = valid & (scores >= thresholds) & (quality >= min_quality)
            sample.gt_instances = instances[keep]
        return batch_data_samples, info

    def loss_by_pseudo_instances(self, batch_inputs, batch_data_samples, batch_info=None):
        base_losses = super().loss_by_pseudo_instances(batch_inputs, batch_data_samples, batch_info)
        cfg = self.semi_train_cfg
        if not cfg.get("enable_consistency", False):
            return base_losses

        bboxes_list = [sample.gt_instances.bboxes for sample in batch_data_samples]
        num_instances = sum(len(b) for b in bboxes_list)
        if num_instances == 0:
            return base_losses

        gt = [sample.gt_instances for sample in batch_data_samples]
        labels = torch.cat([x.labels for x in gt], dim=0)
        scores = torch.cat([x.scores for x in gt], dim=0)
        uncertainties = torch.cat([x.reg_uncs for x in gt], dim=0)
        # Boxes already projected into the student view by SemiBaseDetector.
        rois = bbox2roi(bboxes_list)
        features = self.student.extract_feat(batch_inputs)
        outputs = self.student.roi_head._bbox_forward(features, rois)
        logits, delta = outputs["cls_score"], outputs["bbox_pred"]
        if delta is None:
            raise RuntimeError("ROI head regression logits required")
        coder = self.student.roi_head.bbox_head.bbox_coder
        decoded = coder.decode(rois[:, 1:], delta)
        if decoded.shape[-1] == 4:
            boxes = decoded
        else:
            num_box_classes = decoded.shape[-1] // 4
            if bool((labels >= num_box_classes).any()):
                raise ValueError("bbox class index exceeds regression channels")
            boxes = decoded.reshape(-1, num_box_classes, 4)[
                torch.arange(num_instances, device=labels.device), labels.long()]
        losses = weighted_consistency_loss(
            logits, boxes, torch.cat(bboxes_list, dim=0), labels, scores, uncertainties,
            box_weight=float(cfg.get("consistency_box_weight", 1.0)),
            class_weight=float(cfg.get("consistency_class_weight", 1.0)),
            confidence_weight=float(cfg.get("consistency_confidence_weight", 0.25)),
            reliability_scale=float(cfg.get("reliability_loc_scale", 4.0)))
        lam = float(cfg.get("consistency_weight", 0.5))
        for name, value in losses.items():
            base_losses["unsup_" + name] = value * lam
        return base_losses
