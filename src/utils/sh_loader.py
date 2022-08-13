#Author: Gentry Atkinson
#Organization: Texas University
#Data: 10 August, 2022
#Placeholder data loader for SH Locomotion

import numpy as np

DOWN_SAMPLE = True

if DOWN_SAMPLE:
    path = 'src/data/Sussex_Huawei_DS/'
else:
    path = 'src/data/Sussex_Huawei/'

def sh_loco_load_dataset(incl_xyz_accel=True, incl_rms_accel=False):
    X = np.load(path+'x_train.npy')
    y = np.load(path+'y_train.npy')
    X_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')

    return X, y, X_test, y_test