import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnext101_32x8d(pretrained=False)
        #self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
    
    def forward(self, x):
        x = self.model(x)
        return x