#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 July, 2022
#Translate a signal to a Tensor

from torch import Tensor
import numpy as np

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, signal):
        print("Print type in signal to tensor: ", type(signal))
        print("Len of batch in to tensor: ", len(signal))
        try:
            print("Type of stuff to convert: ", type(signal[0]))
            print("Stuff to convert: ", (signal[0]))
        except:
            pass
        if ' <class \'numpy.ndarray\'>' == type(signal):
            return signal
        else:
            return Tensor(np.array([s for s in signal], dtype=np.float32))