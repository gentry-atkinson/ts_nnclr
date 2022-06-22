#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 June, 2022
#A collection of useful clustering metrics

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import davies_bouldin_score


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

#Silhouette Coefficient
#https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6
#Possible values are -1 to 1, with higher scores being better separation
#SC = (mean nearest cluster distance - mean intracluster distance)/max(intra distance, inter distance)
def sil_coeff(instances, clusters):
    numClusters = len(set(clusters))
    intracluaster_dm = [distance_matrix(instances[np.where(clusters==i)], instances[np.where(clusters==i)]) for i in range(numClusters)]
    intercluster_dm = [distance_matrix(instances[np.where(clusters==i)], instances[np.where(clusters!=i)]) for i in range(numClusters)]
    mean_nearest_cluster = np.mean([np.min(row) for row in intercluster_dm])
    mean_intra_cluster = np.mean(intracluaster_dm)
    return (mean_nearest_cluster - mean_intra_cluster)/np.max([mean_intra_cluster, mean_nearest_cluster])

#Davies Bouldin
#https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
#I'm just stealing this from SK Learn
def db_index(instances, clusters):
    return davies_bouldin_score(instances, clusters)


    

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
    sc = sil_coeff(X, y)
    db = db_index(X, y)
    print(db)