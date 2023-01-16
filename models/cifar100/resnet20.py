import copy
import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from typing import Callable

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels * self.expansion)
            )

        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 16)
        self.relu = nn.ReLU()

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(BasicBlock, 3, out_channels=16, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 3, out_channels=32, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 3, out_channels=64, stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        self.size = self.model_size()

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        """
        - block = same as before, either Basic Block or Bottleneck Block
        - num_residual_blocks = #times it's gonna use the block
        - out_channels = #channels it's gonna be when we're done with that layer
        - stride = the stride, because we halve by 2 every time we have a conv, except in conv2_x where we have stride = 1 
        """
        layers = []
        for i in range(num_residual_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size