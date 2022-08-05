#Author: Gentry Atkinson
#Organization: Texas University
#Data: 05 August, 2022
#Let's analyze NNCLR as a feature extractor for wearable
#  sensor data.

#Experimental Design:
#  Extract features from a HAR dataset using NNCLR, SIMCLR,
#  an auto-encoder, and signal processing from data with added
#  NAR label noise. Compare the P(mispredict) given mislabel to
#  P(mispredict) given correct label

#Hypothesis: NNCLR will be more likely to mispredict a mislabeled
#  instance


run_trad = True
run_ae = True
run_nnclr = True
run_simclr = True

#from utils.import_datasets import get_unimib_data
from load_data_time_series_dev.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
#from load_data_time_series_dev.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from load_data_time_series_dev.HAR.MobiAct.mobiact_adl_load_dataset import mobiact_adl_load_dataset
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import torch

datasets = {
    'unimib' :  tuple(unimib_load_dataset()),
    #'twister' : tuple(e4_load_dataset()),
    #'mobiact' : tuple(mobiact_adl_load_dataset())
}

device = "cuda" if torch.cuda.is_available() else "cpu"

#TensorFlow -> channels last
#PyTorch -> channels first (they get swapped in the feature extractors)

results = {
    '# Mislabeled'      : [],
    'P(mis)'            : [],
    'P(mis|correct)'    : [],
    'P(mis|mislabeled'  : [],
}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, X_test, y_test = datasets[set]

        if X.shape[2] == 1:
            flattened_train = X
        else:
            flattened_train = np.array([np.linalg.norm(i, axis=0) for i in X])

        if X_test.shape[2] == 1:
            flattened_test = X_test
        else:
            flattened_test = np.array([np.linalg.norm(i, axis=0) for i in X_test])

        print('Shape of X: ', X.shape)
        print('Shape of y: ', y.shape)
        print('Shape of flattened train X: ', flattened_train.shape)
        print('Shape of flattened test X: ', flattened_test.shape)
        print('Shape of X_test: ', X_test.shape)
        print('Shape of y_test: ', y_test.shape)