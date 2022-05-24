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
from utils.ts_feature_toolkit import get_features_for_set

if __name__ == '__main__':
    X, y, labels = get_unimib_data()
    X, y = map(np.array, [X, y])
    print('Shape of X: ', X.shape)
    flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X])
    print('Shape of flattened X: ', flattened_X.shape)
    trad_features = get_features_for_set(flattened_X)
    print('Shape of computed features: ', trad_features.shape)