import torch
import torch.nn as nn
import torchvision.models as tvm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class LogisticReg(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, num_classes))
    def forward(self, x): return self.net(x)

def build_model(model_type: str, num_classes=4):
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_type == "logreg":
        return LogisticReg(num_classes)
    if model_type == "resnet18":
        # torchvision 0.24: weights=None корректно отключает предобученные веса
        return tvm.resnet18(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")
