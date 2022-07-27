#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 July, 2022
#Add a small, random variance to each sample

import numpy as np
import torch
from random import gauss, seed

class GaussianNoise(object):
    def __init__(self, prob: float = 0.5, sigma: float = 0.1):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if signal.ndim == 1:
            if np.random.random_sample() < self.prob:
                seed()
                return torch.Tensor([gauss(i, self.sigma) for i in signal])
            else:
                return signal
        else:
            if np.random.random_sample() < self.prob:
                seed()
                return torch.Tensor([[gauss(i, self.sigma) for i in s] for s in signal])
            else:
                return signal