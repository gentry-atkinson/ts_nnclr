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
from matplotlib import pyplot as plt
from utils.import_datasets import get_unimib_data
from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
from utils.ae_feature_learner import get_features_for_set as get_ae_features
from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features
from utils.simclr_feature_learner import get_features_for_set as get_simclr_features

run_trad = False
run_ae = False
run_nnclr = False
run_simclr = True

umap_neighbors = 15
umap_dim = 3

if __name__ == '__main__':
    X, y, labels = get_unimib_data()
    X, y = map(np.array, [X, y])
    print('Shape of X: ', X.shape)
    print('Shape of y: ', y.shape)
    print('Labels in UniMiB: ')
    for l in labels: print(l)
    flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X])
    old_shape = flattened_X.shape
    flattened_X = np.reshape(flattened_X, (old_shape[0], old_shape[1], 1))
    print('Shape of flattened X: ', flattened_X.shape)

    if run_trad:
        trad_features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
        print('Shape of computed features: ', trad_features.shape)
        reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
        embedding = reducer.fit_transform(trad_features)
        print('Shape of UMAP representation: ', embedding.shape)
        plt.figure()
        if umap_dim==2:
            plt.scatter(embedding[:,0], embedding[:,1], c=y)
        else:
            ax = plt.axes(projection ="3d")
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
        plt.show()

    if run_ae:
        ae_features = get_ae_features(flattened_X, with_visual=False)
        print('Shape of autoencoded features: ', ae_features.shape)
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(ae_features)
        print('Shape of UMAP representation: ', embedding.shape)
        plt.figure()
        plt.scatter(embedding[:,0], embedding[:,1], c=y)
        plt.show()

    if run_nnclr:
        nnclr_features = get_nnclr_features(flattened_X, y=y)

    if run_simclr:
        simclr_features = get_simclr_features(flattened_X, y)


