import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

#TODO: Some of the things are not equal to the model definition (from the authors)
# which is here: https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb

#code taken from https://github.com/meliketoy/wide-resnet.pytorch

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_bottleneck_straightned(nn.Module):
    #Bottlebeck because has the structure 1x1 conv, 3x3 conv, 1x1 conv
    #straightned because the dimensions are the same for all convolutions
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_residual=True):
        super(wide_bottleneck_straightned, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        stride3conv = (not use_residual) and stride or 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride3conv, padding=1, bias=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.use_residual = use_residual
        if not self.use_residual:
            self.conv_dim = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=True)

    def forward(self, input):
        x = input
        x = self.dropout1(self.conv1(F.relu(self.bn1(x))))
        x = self.dropout2(self.conv2(F.relu(self.bn2(x))))
        x = self.conv3(F.relu((self.bn3(x))))
        if self.use_residual:
            x += input
        else:
            x += self.conv_dim(input)
        x = F.relu(x)
        return x

class Wide_ResNet_imagenet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_imagenet, self).__init__()

        assert ((depth-5)%12 ==0), 'Wide-resnet depth should be 12n+5'
        n = int((depth-5)/12)
        k = widen_factor

        nStages = [16*k, 32*k, 64*k, 128*k, 256*k]
        self.in_planes = nStages[0]

        self.conv1 = nn.Conv2d(3, nStages[0], 7, stride=2, padding=3, bias=True)
        self.layer1 = self._wide_layer(wide_bottleneck_straightned, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_bottleneck_straightned, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_bottleneck_straightned, nStages[3], n, dropout_rate, stride=2)
        self.layer4 = self._wide_layer(wide_bottleneck_straightned, nStages[4], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[4], momentum=0.9)
        self.linear = nn.Linear(nStages[4], num_classes)
        self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        layers = []

        for idx_block in range(num_blocks):
            use_residual = idx_block != 0
            layers.append(block(self.in_planes, planes, dropout_rate, stride, use_residual))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, kernel_size=7, stride=1, padding=0)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out