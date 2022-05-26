#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
#Let's analyze NNCLR as a feature extractor for wearable
#  sensor data.

#Experimental Design:
#  Extract features from a HAR dataset using NNCLR, SIMCLR,
#  an auto-encoder, and signal processing. Generate UMAP
#  plots to look at. Calculate clustering measures to show
#  separation of classes.

#Hypothesis: NNCLR will produce better separation between classes
#  in the feature space.

import umap
import numpy as np
from utils.import_datasets import get_unimib_data
from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
from utils.ae_feature_learner import get_features_for_set as get_ae_features

run_trad = False
run_ae = False
run_nnclr = True

if __name__ == '__main__':
    X, y, labels = get_unimib_data()
    X, y = map(np.array, [X, y])
    print('Shape of X: ', X.shape)
    flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X])
    old_shape = flattened_X.shape
    flattened_X = np.reshape(flattened_X, (old_shape[0], old_shape[1], 1))
    print('Shape of flattened X: ', flattened_X.shape)

    if run_trad:
        trad_features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
        print('Shape of computed features: ', trad_features.shape)

    if run_ae:
        ae_features = get_ae_features(flattened_X, with_visual=False)
        print('Shape of autoencoded features: ', ae_features.shape)

    if run_nnclr:
        pass


