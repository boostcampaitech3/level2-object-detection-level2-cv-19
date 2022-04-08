# import mmcv
# from mmcv import Config
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)
# from mmdet.models import build_detector
# from mmdet.apis import single_gpu_test
# from mmcv.runner import load_checkpoint
# import os
# from mmcv.parallel import MMDataParallel
# import pandas as pd
# from pandas import DataFrame
# from pycocotools.coco import COCO
# import numpy as np

# # config file 들고오기
# cfg = Config.fromfile('./model.py')

# root='/opt/ml/detection/dataset/'
# epoch = 'best_bbox_mAP_epoch_38'

# # dataset config 수정
# cfg.data.test.test_mode = True

# cfg.data.samples_per_gpu = 32

# cfg.seed=2022
# cfg.gpu_ids = [1]

# # cfg.work_dir
# cfg.work_dir = '/opt/ml/detection/work_dirs/cascadeB_swin_20220404'

# cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
# cfg.model.train_cfg = None
  
# # build dataset & dataloader
# dataset = build_dataset(cfg.data.test)
# data_loader = build_dataloader(
#         dataset,
#         samples_per_gpu=32,
#         workers_per_gpu=cfg.data.workers_per_gpu,
#         dist=False,
#         shuffle=False)

# # checkpoint path
# checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

# model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
# checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

# model.CLASSES = dataset.CLASSES
# model = MMDataParallel(model.cuda(), device_ids=[0])


# output = single_gpu_test(model, data_loader, show_score_thr=0) # output 계산

# # submission 양식에 맞게 output 후처리
# prediction_strings = []
# file_names = []
# coco = COCO(cfg.data.test.ann_file)
# img_ids = coco.getImgIds()

# class_num = 10
# for i, out in enumerate(output):
#     prediction_string = ''
#     image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
#     for j in range(class_num):
#         for o in out[j]:
#             prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
#                 o[2]) + ' ' + str(o[3]) + ' '
        
#     prediction_strings.append(prediction_string)
#     file_names.append(image_info['file_name'])


# submission = pd.DataFrame()
# submission['PredictionString'] = prediction_strings
# submission['image_id'] = file_names
# submission.to_csv(os.path.join(cfg.work_dir, f'cascadeB_swin.csv'), index=None)


import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import pickle

MODEL = ''
PATH = '/opt/ml/detection/work_dirs/cascadeB_swin_small_20220404'
EPOCH = 'best_bbox_mAP_epoch_36.pth'

cfg = Config.fromfile('/opt/ml/detection/baseline/mmdetection/custom_models/cascadeB_swin_small/model.py')

cfg.model.test_cfg.rcnn.max_per_img = 1000

# build dataset & dataloader
cfg.data.test.test_mode = True
size_min = 512
size_max = 1024
multi_scale_list = [(x, x) for x in range(512, 1024+1, 32)]
# multi_scale_list = [(x, x) for x in range(512, 1024+1, 128)]
cfg.data.test.pipeline[1]['img_scale'] = multi_scale_list # Resize
cfg.data.test.ann_file = '/opt/ml/detection/dataset/test.json'

test_dataset = build_dataset(cfg.data.test)
test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# build model

checkpoint_path = os.path.join(PATH, EPOCH)

cfg.optimizer_config.grad_clip = dict(max_norm=35,norm_type=2)
cfg.model.test_cfg.rpn.nms.type='soft_nms'
cfg.model.test_cfg.rcnn.nms.type='soft_nms'
cfg.model.test_cfg.rcnn.score_thr = 0.0

# cfg.model.test_cfg.score_thr=0.0
# cfg.model.test_cfg.nms.type='soft_nms'

# build detector
model = build_detector(cfg.model) 
# ckpt load
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = test_dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, test_loader)

# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()

class_num = 10
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
        
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


submission = pd.DataFrame()
submission['image_id'] = file_names
submission['PredictionString'] = prediction_strings
submission.to_csv(f'cascadeB_swin_20220407_submission_.csv', index=None)