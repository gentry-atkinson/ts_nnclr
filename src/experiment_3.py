#Author: Gentry Atkinson
#Organization: Texas University
#Data: 05 August, 2022
#Let's analyze NNCLR as a feature extractor for wearable
#  sensor data.

#Experimental Design:
#  Extract features from a HAR dataset using NNCLR, SIMCLR,
#  an auto-encoder, and signal processing from data with added
#  NAR label noise. Compare the P(mispredict) given mislabel to
#  P(mispredict) given correct label

#Hypothesis: NNCLR will be more likely to mispredict a mislabeled
#  instance


run_trad = True
run_ae = True
run_nnclr = True
run_simclr = True

#from utils.import_datasets import get_unimib_data
from load_data_time_series_dev.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
#from load_data_time_series_dev.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from load_data_time_series_dev.HAR.MobiAct.mobiact_adl_load_dataset import mobiact_adl_load_dataset
from utils.add_nar import add_nar_from_array
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
#PyTorch -> channels first (they get swapped in the feature extractors)

low_noise_results = {
    'Features'          : [],
    '# Mislabeled'      : [],
    'Acc'               : [],
    'P(mis)'            : [],
    'P(mis|correct)'    : [],
    'P(mis|mislabeled)'  : []
}

high_noise_results = {
    'Features'          : [],
    '# Mislabeled'      : [],
    'Acc'               : [],
    'P(mis)'            : [],
    'P(mis|correct)'    : [],
    'P(mis|mislabeled)'  : []
}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")
        X, y, X_test, y_test = datasets[set]

        if X.shape[2] == 1:
            flattened_train = X
        else:
            flattened_train = np.array([np.linalg.norm(i, axis=0) for i in X])

        if X_test.shape[2] == 1:
            flattened_test = X_test
        else:
            flattened_test = np.array([np.linalg.norm(i, axis=0) for i in X_test])

        y_low_noise_train, low_noise_indexes_train, y_high_noise_train, high_noise_indexes_train = add_nar_from_array(y, len(y[0]))
        y_low_noise_test, low_noise_indexes_test, y_high_noise_test, high_noise_indexes_test = add_nar_from_array(y_test, len(y_test[0]))

        print('Shape of X: ', X.shape)
        print('Shape of y: ', y.shape)
        print('Shape of low noise y: ', y_low_noise_train.shape)
        print('Shape of high noise y: ', y_high_noise_train.shape)
        print("Number low noise train labels altered: ", len(low_noise_indexes_train))
        print("Number high noise train labels altered: ", len(high_noise_indexes_train))
        print('Shape of flattened train X: ', flattened_train.shape)
        print('Shape of flattened test X: ', flattened_test.shape)
        print('Shape of X_test: ', X_test.shape)
        print('Shape of y_test: ', y_test.shape)
        print('Shape of low noise y_test: ', y_low_noise_test.shape)
        print('Shape of high noise y_test: ', y_high_noise_test.shape)
        print("Number low noise test labels altered: ", len(low_noise_indexes_test))
        print("Number high noise test labels altered: ", len(high_noise_indexes_test))

        if(run_trad):        
            from utils.ts_feature_toolkit import get_features_for_set as get_trad_features
            train_features = get_trad_features(np.reshape(flattened_train, (flattened_train.shape[0], flattened_train.shape[1])))
            test_features = get_trad_features(np.reshape(flattened_test, (flattened_test.shape[0], flattened_test.shape[1])))
            print('Shape of Traditional Features: ', train_features.shape)
            #Low Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_low_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print(y_pred)
            print("Trad accuracy low noise: ", accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_low_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            low_noise_results['Features'].append('Traditional')
            low_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            low_noise_results['Acc'].append(accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            low_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            low_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            low_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
            #High Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_high_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("Trad accuracy high noise: ", accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_high_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            high_noise_results['Features'].append('Traditional')
            high_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            high_noise_results['Acc'].append(accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            high_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            high_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            high_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
        if(run_ae):   
            from utils.ae_feature_learner import get_features_for_set as get_ae_features
            train_features, ae_feature_learner = get_ae_features(X, with_visual=False, returnModel=True)
            test_features = ae_feature_learner.predict(X_test)
            print('Shape of AE Features: ', train_features.shape)
            #Low Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_low_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("AE accuracy low noise: ", accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_low_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            low_noise_results['Features'].append('Autoencoder')
            low_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            low_noise_results['Acc'].append(accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            low_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            low_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            low_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
            #High Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_high_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("AE accuracy high noise: ", accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_high_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            high_noise_results['Features'].append('Autoencoder')
            high_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            high_noise_results['Acc'].append(accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            high_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            high_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            high_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
        if(run_nnclr):
            from utils.nnclr_feature_learner import get_features_for_set as get_nnclr_features
            train_features, nnclr_feature_learner = get_nnclr_features(X, y=y, returnModel=True)
            torch_X = torch.tensor(np.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))).to(device)
            torch_X = torch_X.float()
            _, test_features = nnclr_feature_learner(torch_X, return_features=True)
            test_features = test_features.cpu().detach().numpy()
            print('Shape of NNCLR Features: ', train_features.shape)
            #Low Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=3, gamma='scale')
            model.fit(train_features, np.argmax(y_low_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("NNCLR accuracy low noise: ", accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_low_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            low_noise_results['Features'].append('NNCLR')
            low_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            low_noise_results['Acc'].append(accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            low_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            low_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            low_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
            #High Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_high_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("NNCLR accuracy high noise: ", accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_high_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            high_noise_results['Features'].append('NNCLR')
            high_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            high_noise_results['Acc'].append(accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            high_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            high_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            high_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
        if(run_simclr):
            from utils.simclr_feature_learner import get_features_for_set as get_simclr_features
            train_features, simclr_feature_learner = get_simclr_features(X, y=y, returnModel=True)
            torch_X = torch.tensor(np.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))).to(device)
            torch_X = torch_X.float()
            _, test_features = simclr_feature_learner(torch_X, return_features=True)
            test_features = test_features.cpu().detach().numpy()
            print('Shape of SimCLR Features: ', train_features.shape)
            #Low Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=3, gamma='scale')
            model.fit(train_features, np.argmax(y_low_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("SimCLR accuracy low noise: ", accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_low_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_low_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            low_noise_results['Features'].append('SimCLR')
            low_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            low_noise_results['Acc'].append(accuracy_score(np.argmax(y_low_noise_test,axis=-1), y_pred))
            low_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            low_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            low_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)
            #High Noise Labels
            #model = KNeighborsClassifier(n_neighbors=3)
            model = SVC(kernel='poly', degree=5, gamma='scale')
            model.fit(train_features, np.argmax(y_high_noise_train, axis=-1))
            y_pred = model.predict(test_features)
            print("SimCLR accuracy high noise: ", accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            num_correctly_labeled = 0
            num_incorrectly_labeled = 0
            mispred_given_correct_label = 0
            mispred_given_incorrect_label = 0
            for i in range(len(y_test)):
                if np.argmax(y_test[i]) == np.argmax(y_high_noise_test[i]):
                    num_correctly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_correct_label += 1
                else:
                    num_incorrectly_labeled += 1
                    if np.argmax(y_high_noise_test[i]) != y_pred[i]:
                        mispred_given_incorrect_label += 1
            high_noise_results['Features'].append('SimCLR')
            high_noise_results['# Mislabeled'].append(num_incorrectly_labeled)
            high_noise_results['Acc'].append(accuracy_score(np.argmax(y_high_noise_test,axis=-1), y_pred))
            high_noise_results['P(mis)'].append((mispred_given_correct_label+mispred_given_incorrect_label)/(num_correctly_labeled+num_incorrectly_labeled))
            high_noise_results['P(mis|correct)'].append(mispred_given_correct_label/num_correctly_labeled)
            high_noise_results['P(mis|mislabeled)'].append(mispred_given_incorrect_label/num_incorrectly_labeled)

    result_gram = pd.DataFrame.from_dict(low_noise_results)
    result_gram.to_csv('src/results/experiment3_low_noise_dataframe.csv')
    print(result_gram.to_string())

    result_gram = pd.DataFrame.from_dict(high_noise_results)
    result_gram.to_csv('src/results/experiment3_high_noise_dataframe.csv')
    print(result_gram.to_string())