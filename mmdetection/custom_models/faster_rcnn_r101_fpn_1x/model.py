_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
# work_dir = '../../work_dirs/faster_rcnn_r101_fpn_1x_coco'
work_dir = '/opt/ml/detection/work_dirs/faster_rcnn_r101_fpn_1x_coco'