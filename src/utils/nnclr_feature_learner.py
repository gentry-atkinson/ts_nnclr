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


from lightly_plus_time.lightly.models.nnclr import NNCLR
from lightly_plus_time.lightly.models.modules import NNMemoryBankModule
from lightly_plus_time.lightly.data import TS_NNCLRCollateFunction
from lightly_plus_time.lightly.data import LightlyDataset
from lightly_plus_time.lightly.loss import NTXentLoss
from lightly_plus_time.ts_utils.ts_dataloader import UCR2018

def get_features_for_set(X, y=None, with_visual=False, with_summary=False):
    #resnet = torchvision.models.resnet18()
    #backbone = nn.Sequential(*list(resnet.children())[:-1])
    print("Backbone channels in: ", X[0].shape[0])
    backbone = nn.Sequential(
        nn.Conv1d(1, 8, 4, 2, 1, bias=False),
        torch.nn.BatchNorm1d(8),
        torch.nn.ReLU(),
        nn.Conv1d(8, 16, 4, 2, 1, bias=False),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        nn.Flatten()
    )
    model = NNCLR(backbone=backbone, num_ftrs=64, proj_hidden_dim=128, pred_hidden_dim=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    memory_bank = NNMemoryBankModule(size=4096)
    memory_bank.to(device)

    collate_fn = TS_NNCLRCollateFunction(input_size=X[0].shape[0])

    print("X: ", type(X))
    print("y: ", type(y))
    print("X shapre: ", X.shape)
    print("y shape: ", y.shape)

    torch_X = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    #torch_X = torch.utils.data.TensorDataset(np.array([(X[i], y[i]) for i in range(len(X))]))

    dataset = LightlyDataset.from_torch_dataset(torch_X)

    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=16,
      collate_fn=collate_fn,
      shuffle=True,
      drop_last=False,
      num_workers=1,
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    print("Training NNCLR")

    for epoch in range(10):
        total_loss = 0
        #for (x0, x1), _, _ in dataloader:
        for (x0, x1), _ , _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            z0 = memory_bank(z0, update=False)
            z1 = memory_bank(z1, update=True)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    return(np.zeros(X.shape))