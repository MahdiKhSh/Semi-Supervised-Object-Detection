# Research configuration, based on OpenMMLab's official MMDetection 3.x
# SoftTeacher COCO 10% example; requires third_party/mmdetection checkout.
_base_ = ['../third_party/mmdetection/configs/soft_teacher/'
          'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py']

# Category ordering must match prepare_bdd100k.py and COCO category IDs.
classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
data_root = 'data/bdd100k/'
model = dict(detector=dict(roi_head=dict(bbox_head=dict(num_classes=10))))

# Inherited semi dataset uses the correct weak/strong pipelines.
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.update(data_root=data_root, ann_file='annotations/labeled.json',
                       data_prefix=dict(img='images/train/'),
                       metainfo=dict(classes=classes))
unlabeled_dataset.update(data_root=data_root, ann_file='annotations/unlabeled.json',
                         data_prefix=dict(img='images/train/'),
                         metainfo=dict(classes=classes), filter_cfg=dict(filter_empty_gt=False))
train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='GroupMultiSourceSampler', batch_size=2, source_ratio=[1, 1]),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))
val_dataset = _base_.val_dataloader.dataset
val_dataset.update(data_root=data_root, ann_file='annotations/val.json',
                   data_prefix=dict(img='images/val/'),
                   metainfo=dict(classes=classes))
val_dataloader = dict(dataset=val_dataset, batch_size=1, num_workers=2)
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/val.json', metric='bbox')
test_evaluator = val_evaluator

# Short development schedule; NOT an apples-to-apples reproduction of 180k COCO.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=2000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(type='MultiStepLR', begin=0, end=10000, by_epoch=False,
         milestones=[7000, 9000], gamma=0.1)]
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=3))
randomness = dict(seed=42, deterministic=False)
work_dir = 'work_dirs/bdd_softteacher'
