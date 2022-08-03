#Author: Gentry Atkinson
#Organization: Texas University
#Data: 13 July, 2022
#Build and trained a self-supervised feature extractor using Lightly's
#  sim clr

import torch
from torch import nn
#import torchvision

import numpy as np
from random import choice

from torchsummary import summary



from lightly_plus_time.lightly.data import TS_SimCLRCollateFunction
from lightly_plus_time.lightly.data import LightlyDataset
from lightly_plus_time.lightly.loss import NTXentLoss
from lightly_plus_time.lightly.models.modules import SimCLRProjectionHead
from lightly_plus_time.lightly.models.simclr import SimCLR

MAX_EPOCHS = 100
PATIENCE = 5

# class SimCLR(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.projection_head = SimCLRProjectionHead(512, 512, 128)

#     def forward(self, x):
#         x = self.backbone(x).flatten(start_dim=1)
#         z = self.projection_head(x)
#         return z

def get_features_for_set(X, y=None, with_visual=False, with_summary=False, returnModel=False):
    #resnet = torchvision.models.resnet18()
    #backbone = nn.Sequential(*list(resnet.children())[:-1])
    print("Swapping to channels first for PyTorch")
    X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))
    y_flat = np.argmax(y, axis=-1)
    print("Backbone channels in: ", X[0].shape[0])
    print("Backbone samples in: ", X[0].shape[1])
    backbone = nn.Sequential(
        nn.Conv1d(in_channels=X[0].shape[0], out_channels=64, kernel_size=8, stride=1, padding='valid', bias=False),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='valid', bias=False),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        nn.Flatten()
    )

    model = SimCLR(backbone, num_ftrs=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    collate_fn = TS_SimCLRCollateFunction(input_size=X[0].shape[0])

    print("X: ", type(X))
    print("y: ", type(y))
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    torch_X = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y_flat))

    dataset = LightlyDataset.from_torch_dataset(torch_X)

    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=16,
      collate_fn=collate_fn,
      shuffle=True,
      drop_last=False,
      num_workers=8,
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    losses = list()

    print("Starting Training")
    for epoch in range(MAX_EPOCHS):
        total_loss = 0
        for (x0, x1), _, _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        #Early Stopping        
        if len(losses) == PATIENCE:
            if avg_loss > max(losses):
                print("Early stop at epoch ", epoch+1)
                break
            else:
                losses = losses[1:]
        losses.append(avg_loss.cpu().item())

    torch_X = torch.tensor(X).to(device)
    
    torch_X = torch_X.float()
    _, f = model(torch_X, return_features=True)

    if returnModel:
        return f.cpu().detach().numpy(), model
    else:
        return f.cpu().detach().numpy()




