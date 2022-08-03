#Author: Gentry Atkinson
#Organization: Texas University
#Data: 01 August, 2022
#Let's analyze NNCLR as a feature extractor for wearable
#  sensor data.

#Experimental Design:
#  Extract features from a HAR dataset using NNCLR, SIMCLR,
#  an auto-encoder, and signal processing. Use KNN to classify
#  each instance of data in the set

#Hypothesis: NNCLR will have the highest accuracy and F1 when
#  classifying the extracted features

run_trad = True
run_ae = True
run_nnclr = True
run_simclr = True

#from utils.import_datasets import get_unimib_data
from unittest import result
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from load_data_time_series_dev.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
#from load_data_time_series_dev.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from load_data_time_series_dev.HAR.MobiAct.mobiact_adl_load_dataset import mobiact_adl_load_dataset
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
    'Acc'   : [],
    'F1'    : [],
    'Prec'  : [],
    'Rec'   : []
}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, X_test, y_test = datasets[set]
        if X_test.shape[2] == 1:
            flattened_X = X_test
        else:
            flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X_test])   
        print('Shape of X: ', X.shape)
        print('Shape of y: ', y.shape)
        print('Shape of flattened X: ', flattened_X.shape)
        print('Shape of X_test: ', X_test.shape)
        print('Shape of y_test: ', y_test.shape)
        if(run_trad):        
            from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
            trad_features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
            print('Shape of Traditional Features: ', trad_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(trad_features, y_test)
            y_pred = model.predict(trad_features)
            print("Trad accuracy: ", accuracy_score(y_test, y_pred))
            results['Features'].append('Traditional')
            results['Acc'].append(accuracy_score(y_test, y_pred))
            results['F1'].append(f1_score(y_test, y_pred, average='weighted'))
            results['Prec'].append(precision_score(y_test, y_pred, average='weighted'))
            results['Rec'].append(recall_score(y_test, y_pred, average='weighted'))
        if(run_ae):        
            from utils.ae_feature_learner import get_features_for_set as get_ae_features
            _, ae_feature_learner = get_ae_features(X, with_visual=False, returnModel=True)
            ae_features = ae_feature_learner.predict(X_test)
            print('Shape of AE Features: ', ae_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(ae_features, y_test)
            y_pred = model.predict(ae_features)
            print("AE accuracy: ", accuracy_score(y_test, y_pred))
            results['Features'].append('Autoencoder')
            results['Acc'].append(accuracy_score(y_test, y_pred))
            results['F1'].append(f1_score(y_test, y_pred, average='weighted'))
            results['Prec'].append(precision_score(y_test, y_pred, average='weighted'))
            results['Rec'].append(recall_score(y_test, y_pred, average='weighted'))
        if(run_nnclr):
            from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features
            _, nnclr_feature_learner = get_nnclr_features(X, y=y, returnModel=True)
            torch_X = torch.tensor(np.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))).to(device)
            torch_X = torch_X.float()
            _, nnclr_features = nnclr_feature_learner(torch_X, return_features=True)
            nnclr_features = nnclr_features.cpu().detach().numpy()
            print('Shape of NNCLR Features: ', nnclr_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(nnclr_features, y_test)
            y_pred = model.predict(nnclr_features)
            print("NNCLR accuracy: ", accuracy_score(y_test, y_pred))
            results['Features'].append('NNCLR')
            results['Acc'].append(accuracy_score(y_test, y_pred))
            results['F1'].append(f1_score(y_test, y_pred, average='weighted'))
            results['Prec'].append(precision_score(y_test, y_pred, average='weighted'))
            results['Rec'].append(recall_score(y_test, y_pred, average='weighted'))

        if(run_simclr):
            from utils.simclr_feature_learner import get_features_for_set as get_simclr_features
            _, simclr_feature_learner = get_simclr_features(X, y=y, returnModel=True)
            torch_X = torch.tensor(np.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))).to(device)
            torch_X = torch_X.float()
            _, simclr_features = simclr_feature_learner(torch_X, return_features=True)
            simclr_features = simclr_features.cpu().detach().numpy()
            print('Shape of SimCLR Features: ', simclr_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(simclr_features, y_test)
            y_pred = model.predict(simclr_features)
            print("SimCLR accuracy: ", accuracy_score(y_test, y_pred))
            results['Features'].append('SimClr')
            results['Acc'].append(accuracy_score(y_test, y_pred))
            results['F1'].append(f1_score(y_test, y_pred, average='weighted'))
            results['Prec'].append(precision_score(y_test, y_pred, average='weighted'))
            results['Rec'].append(recall_score(y_test, y_pred, average='weighted'))

        result_gram = pd.DataFrame.from_dict(results)
        result_gram.to_csv('src/results/experiment1_dataframe.csv')
        print(result_gram.to_string())