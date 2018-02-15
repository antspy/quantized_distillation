import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, k=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, math.floor(64*k), kernel_size=11, stride=4, padding=2), #originally 64 filters
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(math.floor(64*k), math.floor(192*k), kernel_size=5, padding=2), #originally 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(math.floor(192*k), math.floor(384*k), kernel_size=3, padding=1), #originally 384
            nn.ReLU(inplace=True),
            nn.Conv2d(math.floor(384*k), math.floor(256*k), kernel_size=3, padding=1), #originally 256
            nn.ReLU(inplace=True),
            nn.Conv2d(math.floor(256*k), math.floor(256*k), kernel_size=3, padding=1), #originally 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(math.floor(256*k) * 6 * 6, 4096), #originally 256 * 6 * 6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 6 * 6) #originally 256 * 6 * 6
        x = self.classifier(x)
        return x