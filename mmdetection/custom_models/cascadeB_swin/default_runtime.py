dist_params = dict(backend='nccl')

checkpoint_config = dict(interval=9)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs=dict(
                project='Object Detection',
                entity='next_level',
                name='cascadeB_swin_20220404'
            ),
        )
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', metric=['bbox'])
runner = dict(type='EpochBasedRunner', max_epochs=40)
work_dir = '/opt/ml/detection/work_dirs/cascadeB_swin_20220404'
gpu_ids = [0]
seed = 2022