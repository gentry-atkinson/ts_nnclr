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


run_trad = True
run_ae = True
run_nnclr = True
run_nnclr_t = True
run_simclr = True
run_simclr_t = True

#from utils.import_datasets import get_unimib_data
from load_data_time_series.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
from load_data_time_series.twristar_dataset_demo import e4_load_dataset
from load_data_time_series.HAR.UCI_HAR.uci_har_load_dataset import uci_har_load_dataset
from utils.sh_loader import sh_loco_load_dataset
#from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from datetime import datetime
from os.path import exists
import numpy as np
import pandas as pd
import torch
import json
import gc

datasets = {
    'unimib' :  unimib_load_dataset,
    'twister' : e4_load_dataset,
    'uci har' : uci_har_load_dataset,
    'sussex huawei' : sh_loco_load_dataset
}

device = "cuda" if torch.cuda.is_available() else "cpu"

#TensorFlow -> channels last
#PyTorch -> channels first (they get swapped in the feature extractors)

results = {
    'Features'        : [],
    'Data'            : [],
    'Avg Inter-Dist'  : [],
    'Avg Intra-Dist'  : [],
    'Max Dist'        : [],
    'Min Dist'        : [],
    'Silhouette'      : []
}

raw_distances = {}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, X_test, y_test = datasets[set](incl_xyz_accel=True, incl_rms_accel=False)

        X_total = np.concatenate((X, X_test), axis=0)
        y_total = np.concatenate((y, y_test), axis=0)
        num_labels = len(y_total[0])
        y_flat = np.argmax(y_total, -1)

        if X_total.shape[2] == 1:
            flattened_X = X_total
        else:
            flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X_total])


        print('Shape of X_total: ', X_total.shape)
        print('Shape of y_total: ', y_total.shape)
        print('Shape of flattened X: ', flattened_X.shape)

        if(run_trad):
            if exists('src/features/trad_total_'+set+'.npy'):
                features = np.load('src/features/trad_total_'+set+'.npy')
            else:        
                from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
                features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
                np.save('src/features/trad_total_'+set+'.npy', features)
            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                #print("Label: ", l)
                w = np.where(y_flat==l)
                #print("number of labels: ", len(w))
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('Traditional')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['Traditional '+set] = dist_mat
        
        if(run_ae):
            if exists('src/features/ae_total_'+set+'.npy'):
                features = np.load('src/features/ae_total_'+set+'.npy')
            else:        
                from utils.ae_feature_learner import get_features_for_set as get_ae_features
                features = get_ae_features(X_total, with_visual=False, returnModel=False)
                np.save('src/features/ae_total_'+set+'.npy', features)
            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                w = np.where(y_flat==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('AutoEncoder')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['AE '+set] = dist_mat

        if(run_nnclr):
            if exists('src/features/nnclr_total_'+set+'.npy'):
                features = np.load('src/features/nnclr_total_'+set+'.npy')
            else:
                from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features    
                features = get_nnclr_features(X_total, y=y_total, returnModel=False, bb='CNN')
                np.save('src/features/nnclr_total_'+set+'.npy', features)
            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                w = np.where(y_flat==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('NNCLR')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['NNCLR '+set] = dist_mat

        if(run_nnclr_t):
            if exists('src/features/nnclrt_total_'+set+'.npy'):
                features = np.load('src/features/nnclrt_total_'+set+'.npy')
            else:
                from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_t_features
                features = get_nnclr_t_features(X_total, y=y_total, returnModel=False, bb='Transformer')
                np.save('src/features/nnclrt_total_'+set+'.npy', features)
            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                w = np.where(y_flat==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('NNCLR+T')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['NNCLR+T '+set] = dist_mat

        if(run_simclr):
            if exists('src/features/simclr_total_'+set+'.npy'):
                features = np.load('src/features/simclr_total_'+set+'.npy')
            else:
                from utils.simclr_feature_learner import get_features_for_set as get_simclr_features
                features = get_simclr_features(X_total, y=y_total, returnModel=False, bb='CNN')
                np.save('src/features/simclr_total_'+set+'.npy', features)

            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                w = np.where(y_flat==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('SimCLR')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['SimCLR '+set] = dist_mat

        if(run_simclr_t):
            if exists('src/features/simclrt_total_'+set+'.npy'):
                features = np.load('src/features/simclrt_total_'+set+'.npy')
            else:
                from utils.simclr_feature_learner import get_features_for_set as get_simclr_t_features
                features = get_simclr_t_features(X_total, y=y_total, returnModel=False, bb='Transformer')
                np.save('src/features/simclrt_total_'+set+'.npy', features)

            features_split = []
            dist_mat = []
            gc.collect()
            for l in range(num_labels):
                w = np.where(y_flat==l)
                features_split.append(np.array(features[w][:]))
                print("Instances with label ", l, " : ", len(features_split[l]))
            features_split = np.array(features_split)
            print("Shape of feature split: ", features_split.shape)
            # print(np.mean(cdist(features_split[0], features_split[1], wasserstein_distance)))
            for i in features_split:
                dist_mat.append([np.mean(cdist(i, j, 'cosine')) for j in features_split])
            dist_mat = np.array(dist_mat)
            inter_sum = 0
            intra_sum = 0
            for i in range(len(dist_mat)):
                for j in range(len(dist_mat[0])):
                    if i == j:
                        intra_sum += dist_mat[i][j]
                    else:
                        inter_sum += dist_mat[i][j]
            results['Features'].append('SimCLR+T')
            results['Data'].append(set)
            results['Avg Inter-Dist'].append(inter_sum/(num_labels**2 - num_labels))
            results['Avg Intra-Dist'].append(intra_sum/num_labels)
            results['Max Dist'].append(np.amax(dist_mat))
            results['Min Dist'].append(np.amin(dist_mat))
            results['Silhouette'].append((inter_sum/(num_labels**2 - num_labels))/(intra_sum/num_labels))

            raw_distances['SimCLR '+set] = dist_mat

    result_gram = pd.DataFrame.from_dict(results)
    result_gram.to_csv('src/results/experiment2_dataframe_{}.csv'.format(str(datetime.now())))

    # with open('src/results/experiment_2_raw_distances.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(raw_distances))
    np.save('src/results/experiment_2_raw_distances.txt', raw_distances)
    
    print(result_gram.to_string())