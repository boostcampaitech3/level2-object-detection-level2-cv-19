checkpoint_config = dict(max_keep_ckpts=2, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs=dict(
                project='Object Detection',
                entity='next_level',
                name=''
            ),
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

gpu_ids = [0]
seed = 2022

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50', classwise=True)