import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(2, out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, out_channels * self.expansion)
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

class ResNet(nn.Module):
    """
    - block = Basic Block / Bottleneck Block, in this case for a ResNet-20 we simply have a Basic Block
    - layers = a list which tells us how many times we want to use the Basic Block (look at Table 1 in the paper)
        f.e. in our case for ResNet-20 will be [3,2,2,2]
    - image channels = #channels of the input, for RGB it's 3
    - num_classes = 100 because we use CIFAR100
    """
    def __init__(self, block, layers, image_channels, num_classes=100):
        super(ResNet, self).__init__()
        # Here we're just defining the initial layers and the relu
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(2, 16)
        self.relu = nn.ReLU()

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], out_channels=16, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=32, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=64, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

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

def resnet20(image_channels=3):
    return ResNet(BasicBlock, [3, 3, 3], image_channels)