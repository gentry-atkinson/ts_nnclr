#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 July, 2022
#Translate a signal up, down, left, or right

import numpy as np
import torch

class AmplitudeShift(object):
    def __init__(self, prob: float = 0.5, shift: float = 0.5):
        self.prob = prob
        self.shift = shift

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if np.random.random_sample() < self.prob:
            return torch.Tensor([i+self.shift for i in signal])
        else:
            return signal