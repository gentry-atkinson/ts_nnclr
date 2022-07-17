#Author: Gentry Atkinson
#Organization: Texas University
#Data: 13 July, 2022
#Build and trained a self-supervised feature extractor using Lightly's
#  nearest neighbor clr

import torch
from torch import nn
#import torchvision

import numpy as np
from random import choice

import sys 



# try:
#     from utils.gen_ts_data import generate_pattern_data_as_array
# except:
#     from gen_ts_data import generate_pattern_data_as_array


from lightly_plus_time.lightly.models.nnclr import NNCLR

from lightly_plus_time.lightly.models.modules import NNMemoryBankModule

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    memory_bank = NNMemoryBankModule(size=4096)
    memory_bank.to(device)

    return(np.zeros(X.shape))

# if __name__ == '__main__':
#   print('Verifying NNCLR')
#   X = np.array([
#     generate_pattern_data_as_array(128) for _ in range(100)
#   ])
  
#   X = np.reshape(X, (100,128,1))
#   y = np.array([choice([0,1]) for _ in range(100)])
#   print(X.shape)

#   encoded_X = get_features_for_set(X, y)
#   print(encoded_X.shape)