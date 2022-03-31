# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip= dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2400,
    warmup_ratio=0.001,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
work_dir = '../../work_dirs/swin-t'