from torch import nn
import torchvision
import torch

class Res18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrain_model = torchvision.models.resnet18(pretrained=True)
        self.pretrain_model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True) # resnet18.fc의 in_features의 크기는?

    def forward(self, x):
        x = self.pretrain_model.forward(x)
        return x