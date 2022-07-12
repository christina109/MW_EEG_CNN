# train a meta-learner with outputs from each input_type x channel model


testOn = False
gpu2use = int(input('Which GPU? '))

import sys
if testOn:
    resp = input('Test mode is ON. Continue?[y/n]')
    if not resp.lower() == 'y': sys.exit()

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu2use)

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from keras.models import load_model, model_from_json
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
import keras
import numpy as np
import pandas as pd
import os
import code_pyfile.loadData1 as ld1
import cupy as cp
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from datetime import datetime
import pickle
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from code_pyfile.channel import Channel # add noise adaptation layers


t0 = time.perf_counter()

seeds = [36]
f_stacking = os.path.join('history', 'metalearner_full')
stacking_model = input('Name of meta-learner? ')
f_input =  os.path.join('prediction', 'simplecnn_6_omni') # predictions from sub models

f_output = input('Output folder? ')
p_save = os.path.join('prediction', 'metapred_asp', f_output)

if not os.path.exists(p_save):
    os.makedirs(p_save)
    print('Create output directory')
else:
    if testOn: (print('Will overwrite...'))
    else:     
        resp = input('Output directory EXISTS. Overwrite?[y/n]')
        if not resp.lower() == 'y': sys.exit()
print('Output directory: {}'.format(p_save))

input_models = {#'raw_normChan_cw1.6': {'channel': range(32)},
               # 'power_normChanfreq_cw1.2': {'channel': range(32)},
               # 'ispc_normChanfreq_cw1.6': {'channel': range(120)},
                'wst_normScale_cw1.4': {'channel': range(32)}}


val_indices = [0]
nClass = 2    

# print settings
df_ml = pd.DataFrame({'parameter':['input_models','f_stacking', 'f_input',
                                   'seeds', 'val_indices', 'stacking_model'],
                      'value':[input_models, f_stacking, f_input, 
                               seeds, val_indices, stacking_model]})
df_ml.to_csv(os.path.join(p_save, 'settings.csv'), index = False)

print('############ current setting ############')
print(df_ml)
print('#########################################')
resp = input('Please check the setting. Continue?[y/n]')
if not resp.lower() == 'y': sys.exit()


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


###   MAIN   ###
y_test_all = []  # to summarise performance through validation
yhat_test_all = []
for seed in tqdm(seeds):
    for vali in tqdm(val_indices):

        y_test = []
        X_test = []

        for input_model in tqdm(input_models):

            print('Loading data...')
            p_input= os.path.join(f_input, str(seed))
                                   
            channels = input_models[input_model]['channel']
                     
            for channel in channels:

                f_test_x = os.path.join(p_input, input_model, str(channel),
                                       'pred_val_lnpo{}.txt'.format(vali))

                f_test_y = os.path.join(p_input, input_model, str(channel),
                                       'y_val_lnpo{}.txt'.format(vali))
                
                pred_test = np.genfromtxt(f_test_x, delimiter = ',')
                y_test_tp = np.genfromtxt(f_test_y, delimiter = ',')

                if isinstance(X_test, list):
                    X_test = pred_test.copy()
                else:
                    X_test = np.concatenate((X_test, pred_test), axis = 1)

                if isinstance(y_test, list):
                    y_test = y_test_tp.copy()

                    if not y_test.shape[0] == X_test.shape[0]:
                        print('UNMATCHED X/Y. Abort session.')
                        sys.exit()
                else:
                    if not np.array_equal(y_test, y_test_tp):
                         print('UNMATCHED Y between input models. Abort session.')
                         sys.exit()

                     
        ### meta-prediction  ###
    
        if stacking_model[:2] == 'lr':

            model2load = os.path.join(f_stacking, stacking_model, 'metalearn_seed{}_validation{}.pkl'.format(seed,vali))
            with open(model2load, 'rb') as lf:
                model = pickle.load(lf)
            yhat_test = model.predict(X_test)
                                     

        elif stacking_model[:2] == 'nn':

            model2load = os.path.join(f_stacking, stacking_model, 'metalearn_seed{}_validation{}.h5'.format(seed,vali))
            model = load_model(model2load)
            yhat_test = model.predict(X_test)
            yhat_test = np.argmax(yhat_test, axis = -1)
        
        
        # eval performance
        df_test = eval_performance(yhat_test, y_test, nClass)
        f_test = plot_roc(yhat_test, y_test)
        df_test.to_csv(os.path.join(p_save, 'metrics_test_seed{}_valiation{}.csv'.format(seed, vali)))
        f_test.savefig(os.path.join(p_save, 'roc_test_seed{}_validation{}.png'.format(seed,vali)),
                        dpi = 300, format = 'png')

        # data for overall assessment
        if isinstance(y_test_all, list):
            y_test_all = y_test.copy()
            yhat_test_all = yhat_test.copy()
        else:
            y_test_all = np.concatenate((y_test_all, y_test), axis=0)
            yhat_test_all = np.concatenate((yhat_test_all, yhat_test), axis=0)
            
# overall assessment
df_test_all = eval_performance(yhat_test_all, y_test_all, nClass)
f_test_all = plot_roc(yhat_test_all, y_test_all)
df_test_all.to_csv(os.path.join(p_save, 'metrics_test.csv'))
f_test_all.savefig(os.path.join(p_save, 'roc_test.png'),
                    dpi = 300, format = 'png')

t2 = time.perf_counter()
print("Time elapsed [min]: ", (t2-t0)/60) # CPU seconds elapsed (floating point)


