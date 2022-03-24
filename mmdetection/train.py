from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)



############################################
config_dir_1 = 'faster_rcnn'
config_dir_2 = 'faster_rcnn_r50_fpn_1x_coco.py'
############################################



classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile(f'./configs/{config_dir_1}/{config_dir_2}')

root='/opt/ml/detection/dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]

# work directory 수정
cfg.work_dir = f'./work_dirs/{config_dir_2[:-3]}_trash'

cfg.model.roi_head.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# cfg.runner['max_epochs'] = 24

# Wandb Experiment name 수정
cfg.log_config.hooks[1].init_kwargs['name'] = config_dir_2[:-3]

datasets = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model)
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=False)