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

run_trad = False
run_ae = False
run_nnclr = True
run_simclr = True

from utils.import_datasets import get_unimib_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

datasets = {
    'unimib' :  tuple(get_unimib_data())
}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, labels = datasets[set]
        flattened_X = np.array([np.linalg.norm(i, axis=0) for i in X])   
        print('Shape of X: ', X.shape)
        print('Shape of y: ', y.shape)
        print('Labels in Set: ', ' '.join([str(i) for i in labels]))
        print('Shape of flattened X: ', flattened_X.shape)
        if(run_trad):        
            from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
            trad_features = get_trad_features(np.reshape(flattened_X, (flattened_X.shape[0], flattened_X.shape[1])))
            print('Shape of Traditional Features: ', trad_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(trad_features, y)
            y_pred = model.predict(trad_features)
            print("Trad accuracy: ", accuracy_score(y, y_pred))
        if(run_ae):        
            from utils.ae_feature_learner import get_features_for_set as get_ae_features
            ae_features = get_ae_features(np.reshape(X, (X.shape[0], X.shape[2], X.shape[1])), with_visual=False)
            print('Shape of AE Features: ', ae_features.shape)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(ae_features, y)
            y_pred = model.predict(ae_features)
            print("AE accuracy: ", accuracy_score(y, y_pred))
        if(run_nnclr):
            from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features
            nnclr_features = get_nnclr_features(np.reshape(X, (X.shape[0], X.shape[2], X.shape[1])), y=y)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(nnclr_features, y)
            y_pred = model.predict(nnclr_features)
            print("NNCLR accuracy: ", accuracy_score(y, y_pred))
