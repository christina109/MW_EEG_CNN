# code to replicate model by Hosseini & Guo (2019)
# reference: Hosseini, S., & Guo, X. (2019). Deep Convolutional Neural Network for Automated Detection of Mind Wandering using EEG Signals. Paper presented at the Proceedings of the 10th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics. 
# script correspondance: cyj.sciATgmail.com

testOn = False  # f_hist will be a temporary folder
# device number
gpu2use = input('Please specify the gpu to use: ')
gpu2use = int(gpu2use)

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

from sklearn.metrics import classification_report, roc_curve, auc
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
import code_pyfile.loadData1_Hosseini2019 as ld1
import code_pyfile.loadData2_Hosseini2019 as ld2
from itertools import compress, cycle
import time
from tqdm import tqdm
from collections import Counter
import pickle as pk
import matplotlib.pyplot as plt
from code_pyfile.channel import Channel # add noise adaptation layers

t0 = time.perf_counter()


###   settings   ###

# randomization
randseed = 36  # [36, 54, 87]  # 27 during testing phase

# dataset/feature/channel
study = 2
input_type = 'raw' # one of ['raw', 'power', 'ispc', 'wst']
norm = 'chan'  # percentage scales the data when loading data. One of ['off', 'chan', 'freq', 'chanfreq', 'trial', 'signal']
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 17 #[[12,12], [10,20]]  # load data from x seconds before each probe
channels = range(32)
subs = range(301,331)  # 1-30 for study 1, 301-330 for study 2

# loading path
neural_network = 'Hosseini2019'
validations = [0]
model_name = input('Specify the model:')
f_load = os.path.join('.', 'history', neural_network, str(randseed), model_name)

# output path
output_name = input('Folder for output? ')
f_save = os.path.join('.', 'prediction', 'Hosseini2019', str(randseed), output_name)

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
                                      'channels', 'participants', 'neural_network',
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


# config
nClass = len(conds)

def eval_performance(y_pred, y_true, nClass):
    y_pred = list(y_pred)
    y_true = list(y_true)
    
    # create a confusion matrix
    conf =  np.zeros((nClass, nClass))
    for t,p in zip(y_true, y_pred):
        conf[int(p),int(t)] += 1

    df_perf = pd.DataFrame({'metric': ['sensitivity', 'specificity',
                                       'precision', 'accuracy'],
                            'value': [conf[1,1]/sum(conf[:,1]),
                                      conf[0,0]/sum(conf[:,0]),
                                      conf[1,1]/sum(conf[1,:]),
                                      sum(np.diag(conf))/np.sum(conf)]
                            })
    if testOn: print(df_perf)
    return df_perf
          

def plot_roc(y_pred,y_true):
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    roc_area = auc(fpr,tpr)

    f = plt.figure()
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr,label='area={:.3f}'.format(roc_area))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    if testOn: plt.show()

    return f


###   modelling   ###
def modelling(x_test, y_test, lnpoi):

    # preprocess
    x_test = np.expand_dims(x_test, 3)
    yc_test = keras.utils.to_categorical(y_test, num_classes=nClass)

    m2load = os.path.join(f_load, 'model_lnpo{}.h5'.format(lnpoi))
    model = load_model(m2load)

    yhat_test = model.predict(x_test)
    yhat_test = np.argmax(yhat_test, axis = -1)

    df_test = eval_performance(yhat_test, y_test, nClass)
    f_test = plot_roc(yhat_test, y_test)

    # save
    df_test.to_csv(os.path.join(f_save, 'metrics_test_seed{}_valiation{}.csv'.format(randseed, lnpoi)))
    f_test.savefig(os.path.join(f_save, 'roc_test_seed{}_validation{}.png'.format(randseed,lnpoi)),
                  dpi = 300, format = 'png')

    return yhat_test
        


def lnpocv(channels, norm=norm, winlen=win_lens, subs=subs):
    # load all datasets
    if study == 1:
        x_all, y_all, s_all = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [channels], gpu2use)
    else:
        x_all, y_all, s_all = ld2.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [channels], gpu2use)

    # to summarise performance through validation
    y_test_all = []
    yhat_test_all = []
    for lnpoi in tqdm(validations):
        
        if gpu2use > -1:
            print('Cupy array conversion...')
            x_test = cp.asnumpy(x_all)
            y_test = cp.asnumpy(y_all)
    
        yhat_test = modelling(x_test, y_test, lnpoi)

        if isinstance(y_test_all, list):
            y_test_all = y_test.copy()
            yhat_test_all = yhat_test.copy()
        else:
            y_test_all = np.concatenate((y_test_all, y_test), axis=0)
            yhat_test_all = np.concatenate((yhat_test_all, yhat_test), axis=0)
            
    # overall assessment
    df_test_all = eval_performance(yhat_test_all, y_test_all, nClass)
    f_test_all = plot_roc(yhat_test_all, y_test_all)
    
    df_test_all.to_csv(os.path.join(f_save, 'metrics_test.csv'))
    f_test_all.savefig(os.path.join(f_save, 'roc_test.png'),
                      dpi = 300, format = 'png')
        

        
# main function
lnpocv(channels = channels)

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)
