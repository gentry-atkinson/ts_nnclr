#Author: Gentry Atkinson
#Organization: Texas University
#Data: 13 July, 2022
#Build and trained a self-supervised feature extractor using Lightly's
#  nearest neighbor clr

from lightly_plus_time.lightly.models.nnclr import NNCLR

import torch
from torch import nn
import torchvision

def get_features_for_set(X, y=None, with_visual=False, with_summary=False):
    #resnet = torchvision.models.resnet18()
    #backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = nn.Sequential(
          nn.Conv1d(in_channels=X[0].shape[1], out_channels=128, kernel_size=16, ),
          nn.ReLU(),
          nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, ),
          nn.ReLU()
        )
    model = NNCLR(backbone)