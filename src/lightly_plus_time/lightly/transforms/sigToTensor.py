#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 July, 2022
#Translate a signal to a Tensor

from torch import Tensor

class AmplitudeShift(object):
    def __init__(self):
        pass

    def __call__(self, signal):
        return Tensor(signal)