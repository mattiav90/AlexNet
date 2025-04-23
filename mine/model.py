import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import torch.optim as optim
from torch.quantization import FakeQuantize, default_qconfig, QConfig
from torch.quantization.observer import MovingAverageMinMaxObserver
import matplotlib.pyplot as plt
from conf import conf



# Optional: 4-bit FakeQuantize config
# def get_4bit_qconfig():
#     return QConfig(
#         activation=FakeQuantize.with_args(
#             observer=MovingAverageMinMaxObserver,
#             quant_min=0, quant_max=15, dtype=torch.quint8
#         ),
#         weight=FakeQuantize.with_args(
#             observer=MovingAverageMinMaxObserver,
#             quant_min=-8, quant_max=7, dtype=torch.qint8
#         )
#     )


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, qat=False):
        super(AlexNet, self).__init__()
        self.quant_aware_train = qat

        # Quantization placeholders
        if self.quant_aware_train:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),   #0
            nn.ReLU(inplace=True),                                  #1
            nn.MaxPool2d(kernel_size=2, stride=2),                  #2

            nn.Conv2d(64, 192, kernel_size=3, padding=1),           #3
            nn.ReLU(inplace=True),                                  #4
            nn.MaxPool2d(kernel_size=2, stride=2),                  #5

            nn.Conv2d(192, 384, kernel_size=3, padding=1),          #6
            nn.ReLU(inplace=True),                                  #7

            nn.Conv2d(384, 256, kernel_size=3, padding=1),          #8
            nn.ReLU(inplace=True),                                  #9

            nn.Conv2d(256, 256, kernel_size=3, padding=1),          #10
            nn.ReLU(inplace=True),                                  #11
            nn.MaxPool2d(kernel_size=2, stride=2),                  #12
        )

        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        if self.quant_aware_train:
            x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.quant_aware_train:
            x = self.dequant(x)
        return x

    # Fusion of Conv2d and ReLU layers for QAT
    def fuse_model(self):
        torch.quantization.fuse_modules(self.features, 
                                        [['0', '1'], ['3', '4'], ['6', '7'], ['8', '9'], ['10', '11']], inplace=True)

    # Evaluation function
    def evaluate(self, dataloader, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy