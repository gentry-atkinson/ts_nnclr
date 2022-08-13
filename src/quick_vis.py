#Author: Gentry Atkinson
#Organization: Texas University
#Data: 13 August, 2022
#Visualize some feature sets

import os

import umap.umap_ as umap
import numpy as np
from load_data_time_series.twristar_dataset_demo import e4_load_dataset
from matplotlib import pyplot as plt

DATASET = 'twister'
umap_neighbors = 15

if __name__ == '__main__':
    file_list = os.listdir('src/features')
    print(file_list)
    X, y, X_test, y_test = e4_load_dataset(incl_xyz_accel=True, incl_rms_accel=False)
    y = np.argmax(y, axis=-1)
    for f in file_list:
        
        if DATASET in f and 'train' in f:
            title = f[:-4]
            plt.figure()
            plt.title(title)
            features = np.load('src/features/'+f)
            reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=3)
            embedding = reducer.fit_transform(features)
            print('Shape of X: ', X.shape)
            print('Shape of y: ', y.shape)
            print('Shape of UMAP representation: ', embedding.shape)
            ax = plt.axes(projection ="3d")
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
            plt.savefig('imgs/'+title+'.pdf')