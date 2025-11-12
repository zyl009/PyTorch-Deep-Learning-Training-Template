import torch
import torch.nn as nn
import torchvision.models as models

def create_resnet18(num_classes=10):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
