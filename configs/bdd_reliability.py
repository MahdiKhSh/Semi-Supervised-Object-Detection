# Ablation: identical data/model setup, but enable experimental pseudo-label gate.
_base_ = ['./bdd_softteacher.py']
custom_imports = dict(imports=['ssod_research.mmdet_extension'], allow_failed_imports=False)
model = dict(
    type='ReliabilitySoftTeacher',
    semi_train_cfg=dict(
        enable_reliability=True,
        reliability_loc_scale=4.0,
        reliability_min_score=0.5,
        reliability_min_quality=0.3,
        # Optional thresholds: one entry per BDD category in exact class order.
        reliability_class_thresholds=None))
work_dir = 'work_dirs/bdd_reliability'
