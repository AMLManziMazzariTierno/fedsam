import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class ReTrainModel(nn.Module):

    def __init__(self):
        super(ReTrainModel, self).__init__()

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,100)

    def forward(self, input):

        return self.classifier(input)