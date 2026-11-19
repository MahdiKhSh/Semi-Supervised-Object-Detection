# Supervised-only control on exactly the labeled subset used by SoftTeacher.
# Must be checked against the selected MMDetection revision before training.
_base_ = ['./bdd_softteacher.py']

# Replace teacher/student wrapper with the identical underlying detector.
model = dict(_delete_=True, **_base_.model.detector)
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=_base_.labeled_dataset)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
custom_hooks = []
work_dir = 'work_dirs/bdd_supervised'
