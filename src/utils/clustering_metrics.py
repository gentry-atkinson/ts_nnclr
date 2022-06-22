#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 June, 2022
#A collection of useful clustering metrics

import numpy as np
from scipy.spatial import distance_matrix


#Dunn Index
#https://en.wikipedia.org/wiki/Dunn_index
#DI = min(distance between any two clusters)/max(size of one cluster)
def dunn_index(instances, clusters):
    numClusters = len(set(clusters))
    intercluaster_dm = [distance_matrix(instances[np.where(clusters==i)], instances[np.where(clusters==i)]) for i in range(numClusters)]
    intracluaster_dm = [distance_matrix(instances[np.where(clusters==i)], instances[np.where(clusters!=i)]) for i in range(numClusters)]
    min_dis = np.min(intracluaster_dm)
    max_size = np.max(intercluaster_dm)
    return min_dis/max_size
    
    

if __name__ == '__main__':
    X = np.array([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [0, 1, 2, 4, 5, 6],
        [5, 4, 3, 2, 1, 0],
        [6, 5, 4, 3, 2, 1],
        [6, 4, 3, 2, 1, 0]
    ])
    y = np.array([0,0,0,1,1,1])
    di = dunn_index(X, y)
    print(di)