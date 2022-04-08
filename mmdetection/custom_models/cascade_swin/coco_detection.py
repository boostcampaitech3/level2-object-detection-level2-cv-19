# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
multi_scale = [(x, x) for x in range(512, 1024+1, 32)]
alb_transform = [
    dict(type='VerticalFlip', p=0.15),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='OneOf', transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ], p=0.1),
    dict(type='OneOf', transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ChannelDropout', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0),
        ], p=0.1),
    dict(type='OneOf', transforms=[
        dict(type='ShiftScaleRotate', p=1.0),
        dict(type='RandomRotate90', p=1.0)
    ], p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CutOut',n_holes=10, cutout_ratio=[(0.1, 0.1), (0.05, 0.05)]),
    dict(
        type='Albu', 
        transforms=alb_transform, 
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True
        ),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='AutoAugment',
    #     policies=[[
    #         dict(
    #             type='Resize',
    #             img_scale=multi_scale,
    #             multiscale_mode='value',
    #             keep_ratio=True)
    #     ],
    #               [
    #                   dict(
    #                       type='Resize',
    #                       img_scale=[(300, 512), (350, 512), (400, 512)],
    #                       multiscale_mode='value',
    #                       keep_ratio=True),
    #                   dict(
    #                       type='RandomCrop',
    #                       crop_type='absolute_range',
    #                       crop_size=(576, 800),
    #                       allow_negative_crop=True),
    #                   dict(
    #                       type='Resize',
    #                       img_scale=multi_scale,
    #                       multiscale_mode='value',
    #                       override=True,
    #                       keep_ratio=True)
    #               ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val1.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes))