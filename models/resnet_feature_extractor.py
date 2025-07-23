# ResNet feature extractor for modular use

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import AdaptiveAvgPool2d

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_features=1024):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, output_features)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x 