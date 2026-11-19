# Ablation: identical data/model setup with reliability selector and optional
# differentiable matched-ROI multi-level consistency.
_base_ = ['./bdd_softteacher.py']
custom_imports = dict(imports=['ssod_research.mmdet_extension'], allow_failed_imports=False)
model = dict(
    type='ReliabilitySoftTeacher',
    semi_train_cfg=dict(
        enable_reliability=True,
        reliability_loc_scale=4.0,
        reliability_min_score=0.5,
        reliability_min_quality=0.3,
        reliability_class_thresholds=None,
        enable_consistency=True,
        consistency_weight=0.5,
        consistency_box_weight=1.0,
        consistency_class_weight=1.0,
        consistency_confidence_weight=0.25))
work_dir = 'work_dirs/bdd_reliability'
