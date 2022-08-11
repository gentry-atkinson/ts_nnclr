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

from torchsummary import summary


from lightly_plus_time.lightly.models.nnclr import NNCLR
from lightly_plus_time.lightly.models.modules import NNMemoryBankModule
from lightly_plus_time.lightly.data import TS_NNCLRCollateFunction
from lightly_plus_time.lightly.data import LightlyDataset
from lightly_plus_time.lightly.loss import NTXentLoss
import multiprocessing


MAX_EPOCHS = 5
PATIENCE = 5
NUM_FEATURES = 64

def get_features_for_set(X, y=None, with_visual=False, with_summary=False, bb='CNN', returnModel=False):
    #resnet = torchvision.models.resnet18()
    #backbone = nn.Sequential(*list(resnet.children())[:-1])
    print("Swapping to channels first for PyTorch")
    X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))
    y_flat = np.argmax(y, axis=-1)
    print("Backbone channels in: ", X[0].shape[0])
    print("Backbone samples in: ", X[0].shape[1])
    if bb == 'CNN':
        backbone = nn.Sequential(
            nn.Conv1d(in_channels=X[0].shape[0], out_channels=64, kernel_size=8, stride=1, padding='valid', bias=False),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='valid', bias=False),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            nn.Dropout(p=0.1),
            torch.nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    elif bb == "Transformer":
        backbone = nn.Sequential(
            nn.Conv1d(in_channels=X[0].shape[0], out_channels=64, kernel_size=8, stride=1, padding='valid', bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            nn.LazyLinear(out_features=64),
            torch.nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8) , num_layers=4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    else:
        print("Invalid backbone")
        exit()
    
    model = NNCLR(backbone=backbone, num_ftrs=NUM_FEATURES, proj_hidden_dim=64, pred_hidden_dim=64, )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    memory_bank = NNMemoryBankModule(size=4096)
    memory_bank.to(device)

    print("Backbone ", backbone)
    #print("Backbone: \n", summary(backbone, X[0].shape))

    collate_fn = TS_NNCLRCollateFunction(input_size=X[0].shape[0])

    print("X: ", type(X))
    print("y: ", type(y))
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    torch_X = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y_flat))
    #torch_X = torch.utils.data.TensorDataset(np.array([(X[i], y[i]) for i in range(len(X))]))

    dataset = LightlyDataset.from_torch_dataset(torch_X)

    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=16,
      collate_fn=collate_fn,
      shuffle=True,
      drop_last=False,
      num_workers=multiprocessing.cpu_count(),
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training NNCLR")

    losses = list()

    for epoch in range(MAX_EPOCHS):
        total_loss = 0  
        for (x0, x1), _ , _ in dataloader:
            #print("Type of x0 in main loop: ", type(x0))
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
        print(f"epoch: {epoch+1:>02}, loss: {avg_loss:.5f}")
        #Early Stopping        
        if len(losses) == PATIENCE:
            if avg_loss > max(losses):
                print("Early stop at epoch ", epoch+1)
                break
            else:
                losses = losses[1:]
        losses.append(avg_loss.cpu().item())
        #print(len(losses), ' ', losses)
    model.train(False)
    print("Training finished for NNCLR")    

    del torch_X
    del dataloader

    feat = None
    torch_X = torch.tensor(X)
    torch_X = torch_X.to(device).float()

    # print("Output channels in: ", X[0].shape[0])
    # print("Output samples in: ", X[0].shape[1])

    for signal in torch_X:
        signal = signal.to(device).float()
        signal = torch.reshape(signal, (1, torch_X.shape[1], torch_X.shape[2]))
        (_, _), f = model(signal, return_features=True)
        if feat is None:
            feat = f.cpu().detach().numpy()
        else:
            feat = np.concatenate((feat, f.cpu().detach().numpy()), axis=0)

    # (_,_), feat = model(torch_X, return_features=True)
    # feat = feat.cpu().detach().numpy()

    print("Shape of NNCLR features before return: ", feat.shape)
    

    if returnModel:
        return feat, model
    else:
        return feat