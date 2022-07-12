# across-study prediction with each input models


testOn = False  # f_hist will be a temporary folder
# device number
gpu2use = input('Please specify the gpu to use: ')
gpu2use = int(gpu2use)
#gpu2use = 3  # -1 if cpu

saveModelOn = True

import sys
if testOn:
    resp = input('Test mode is ON. Continue?[y/n]')
    if not resp.lower() == 'y': sys.exit()
             
if not saveModelOn:
    resp = input('SaveModel is OFF. Continue?[y/n]')
    if not resp.lower() == 'y': sys.exit()


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu2use)


from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping
from keras import activations
import keras as keras
from keras.models import model_from_json
from keras.utils import plot_model
import cupy as cp
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import os
import code_pyfile.loadData1 as ld1
import code_pyfile.loadData2 as ld2
from itertools import compress, cycle
import time
from tqdm import tqdm
from collections import Counter
import pickle as pk
from code_pyfile.channel import Channel # add noise adaptation layers

t0 = time.perf_counter()


###   settings   ###

# randomization
randseed = 36  # [36, 54, 87]  # 27 during testing phase

# dataset/feature/channel
study = 2
input_type = input('Input type for testing data? ')
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 17 #[[12,12], [10,20]]  # load data from x seconds before each probe

ci_start = input('Please type the first channel id: ')
ci_end = input('Please type the end channel id (excl.): ')
channels = range(int(ci_start),int(ci_end))
#channels = range(32)
subs = range(301,331) # 1-30 for study 1, 301-330 for study 2

norm = input('Normalization for testing data?')
# percentage scales the data when loading data. One of ['off', 'chan', 'freq', 'chanfreq', 'trial', 'signal'

# loading path
neural_network = 'simplecnn_6_omni'
validations = [0]
model_name = input('Specify the model: ')
f_load = os.path.join('.', 'history', neural_network, str(randseed), model_name)

# output path
f_save = os.path.join('.', 'prediction_win17', neural_network, str(randseed), model_name)


# create path if not existing
if not os.path.exists(f_save):
    os.makedirs(f_save)
    print('Create output directory')
else:
    if testOn: (print('Will overwrite...'))
    else:     
        resp = input('Output directory EXISTS. Overwrite?[y/n]')
        if not resp.lower() == 'y': sys.exit()
print('Output directory: {}'.format(f_save))

# print settings
df_pars = pd.DataFrame({'parameter': ['rand_seed', 'study', 'input_type', 'conditions', 'tasks',
                                      'channels', 'participants','neural_network',
                                      'normalization', 'window_length'],
                        'value':[randseed, study, input_type, conds, tasks,
                                 channels, subs, neural_network, 
                                 norm, win_lens]})
df_pars.to_csv(os.path.join(f_save, 'settings.csv'), index = False)

print('############ current setting ############')
print(df_pars)
print('#########################################')
resp = input('Please check the setting. Continue?[y/n]')
if not resp.lower() == 'y': sys.exit()


###   modelling   ###
def modelling(x_test, y_test, lnpoi, channel):

    # preprocess       
    yc_test = keras.utils.to_categorical(y_test, num_classes=len(conds))

    f_hist = os.path.join(f_save, str(channel))
    if not os.path.exists(f_hist):
        os.makedirs(f_hist)
        print('Create output directory: {}'.format(f_hist))

    m2load = os.path.join(f_load, str(channel), 'model_lnpo{}.h5'.format(lnpoi))
    model = load_model(m2load)

    yhat_test = model.predict(x_test)
    np.savetxt(os.path.join(f_hist, 'pred_val_lnpo{}.txt'.format(lnpoi)), yhat_test, delimiter=',')
    np.savetxt(os.path.join(f_hist, 'y_val_lnpo{}.txt'.format(lnpoi)), y_test, delimiter=',')
    

def lnpocv(channel, norm=norm, winlen=win_lens, subs=subs):
    # load all datasets
    if study == 1:
        x_all, y_all, s_all = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [[channel]], gpu2use)
    else:
        x_all, y_all, s_all = ld2.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [[channel]], gpu2use)

    for lnpoi in tqdm(validations):

        if gpu2use > -1:
            print('Cupy array conversion...')
            x_test = cp.asnumpy(x_all)
            y_test = cp.asnumpy(y_all)
                        
        # batch x (freq x) time 
        if input_type == 'raw':
            x_test = np.transpose(x_test, (0, 2, 1))

        modelling(x_test, y_test, lnpoi, channel)
            

        
# main function
for channel in channels:
    lnpocv(channel = channel)

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)
