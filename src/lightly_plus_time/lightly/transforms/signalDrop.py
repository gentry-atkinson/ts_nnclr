#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2022
#Set some samples in a signal to 0

import numpy as np
import torch

class RandomSignalDrop(object):
    def __init__(self, prob: float = 0.5, drop: float = 0.1):
        self.prob = prob
        self.drop = drop

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if signal.ndim == 1:
            if np.random.random_sample() < self.prob:
                return torch.Tensor([i if np.random.random_sample() >= self.drop else 0 for i in signal[:]])
            else:
                return signal
        else:
            if np.random.random_sample() < self.prob:
                return torch.Tensor([[i if np.random.random_sample() >= self.drop else 0 for i in s[:]] for s in signal])
            else:
                return signal

class WindowedSignalDrop(object):
    def __init__(self, prob: float = 0.5, len: float = 0.1):
        self.prob = prob
        if len > 0.25:
            print("Window length must be <= 0.25 for Windowed Signal Drop")
            self.window_len = 0.25
        else:
            self.window_len = len

    def __call__(self, signal):
        #Set some samples to 0 with random chance
        if np.random.random_sample() < self.prob:
            signalLength = len(signal)
            start = np.random.randint(0, 3*signalLength//4-1)
            stop = start + self.window_len
            if stop >= signalLength: stop = signalLength-1
            return torch.Tensor([0 if i >= start and i <= stop else signal[i] for i in range(signalLength)])
        else:
            return signal

class PeriodicSignalDrop(object):
    def __init__(self, prob: float = 0.5, period: int = 10):
        self.prob = prob
        self.period = period
    def __call__(self, signal):
        if np.random.random_sample() < self.prob:
            return torch.Tensor( [0 if i%self.period==0 else signal[i] for i in range(len(signal))])
        else:
            return signal