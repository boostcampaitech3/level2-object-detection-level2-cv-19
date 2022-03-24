from torch import nn
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrain_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.pretrain_model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 11 # 10 class + background
        self.pretrain_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        x = self.pretrain_model(images, targets)
        return x