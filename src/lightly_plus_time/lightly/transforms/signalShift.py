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

class TimeShift(object):
    def __init__(self, prob: float = 0.5, shift: int = 20):
        self.prob = prob
        self.shift = shift

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        signalLength = len(signal)
        if np.random.random_sample() < self.prob:
            return torch.Tensor([signal[(i+self.shift)%signalLength] for i in range(signalLength)])
        else:
            return signal

class Flip(object):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if np.random.random_sample() < self.prob:
            return np.array([-1*x for x in signal])
        else:
            return signal