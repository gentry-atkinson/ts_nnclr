#Author: Gentry Atkinson
#Organization: Texas University
#Data: 03 August, 2022
#Let's analyze NNCLR as a feature extractor for wearable
#  sensor data.

#Experimental Design:
#  Extract features from a HAR dataset using NNCLR, SIMCLR,
#  an auto-encoder, and signal processing. Measure the mean
#  Wasserstein distance between each class in the feature
#  space

#Hypothesis: NNCLR will have the the highest average distance
#  between classes in the feature space


run_trad = False
run_ae = False
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
#PyTorch -> channels first

results = {
    'Features'  : [],
    'Avg Dist'  : [],
    'Max Dist'  : [],
    'Min Dist'  : [],
}

raw_distances = {}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, X_test, y_test = datasets[set]

        X_total = np.concatenate((X, X_test), axis=0)
        y_total = np.concatenate((y, y_test), axis=0)
        num_labels = len(y_total[0])
        y_total = np.argmax(y_total, -1)

        if X_total.shape[2] == 1:
            flattened_X = X_total
        else:
            flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X_total])


        print('Shape of X_total: ', X_total.shape)
        print('Shape of y_total: ', y_total.shape)
        print('Shape of flattened X: ', flattened_X.shape)

        if(run_trad):        
            from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
            features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
            features_split = []
            dist_mat = []
            for l in range(num_labels):
                #print("Label: ", l)
                w = np.where(y_total==l)
                #print("number of labels: ", len(w))
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j)) for j in features_split])
            dist_mat = np.array(dist_mat)
            results['Features'].append('Traditional')
            results['Avg Dist'].append(np.mean(dist_mat))
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))

            raw_distances['Traditional'] = dist_mat
        
        if(run_ae):        
            from utils.ae_feature_learner import get_features_for_set as get_ae_features
            features = get_ae_features(X_total, with_visual=False, returnModel=False)
            features_split = []
            dist_mat = []
            for l in range(num_labels):
                w = np.where(y_total==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j)) for j in features_split])
            dist_mat = np.array(dist_mat)
            results['Features'].append('AutoEncoder')
            results['Avg Dist'].append(np.mean(dist_mat))
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))

            raw_distances['Traditional'] = dist_mat

        if(run_nnclr):
            print("Shape of X_total: ", X_total.shape)
            from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features
            
            features = get_nnclr_features(X_total, y=y_total, returnModel=False)
            features_split = []
            dist_mat = []
            for l in range(num_labels):
                w = np.where(y_total==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j)) for j in features_split])
            dist_mat = np.array(dist_mat)
            results['Features'].append('AutoEncoder')
            results['Avg Dist'].append(np.mean(dist_mat))
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))

            raw_distances['Traditional'] = dist_mat

    result_gram = pd.DataFrame.from_dict(results)
    result_gram.to_csv('src/results/experiment2_dataframe.csv')
    # for k in raw_distances.keys():
    #     np.array(raw_distances[k]).savetext('src/results/distance_'+k+'.csv', delimeter=',')
    print(result_gram.to_string())