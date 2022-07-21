#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2022
#Set some samples in a signal to 0

import numpy as np
import torch

class RandomSignalDrop(object):

    def __init__(self, prob: float = 0.5, drop: float = 0.1):
        self.prob = prob

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if np.random.random_sample() < self.prob:
            return torch.Tensor([i if np.random.random_sample() >= self.drop else 0 for i in signal[:]])
        else:
            return signal

class WindowedSignalDrop(object):

    def __init__(self, prob: float = 0.5, drop: float = 0.1):
        self.prob = prob

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if np.random.random_sample() < self.prob:
            return torch.Tensor([i if np.random.random_sample() >= self.drop else 0 for i in signal[:]])
        else:
            return signal