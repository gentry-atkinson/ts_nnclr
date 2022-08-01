#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
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

from utils.import_datasets import get_unimib_data

datasets = {
    'unimib' :  tuple(get_unimib_data())
}

if __name__ == '__main__':
    for set in datasets.keys():
        print("------------Set: ", set, "------------")