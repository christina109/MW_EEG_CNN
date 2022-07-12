
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

from sklearn.metrics import classification_report, roc_curve, auc
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers import DepthwiseConv2D, AveragePooling2D, SeparableConv2D
from keras.regularizers import l2, l1
from tensorflow.keras.constraints import max_norm
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
import matplotlib.pyplot as plt

t0 = time.perf_counter()


###   settings   ###

# randomization
randseed = 36  # [36, 54, 87]  # 27 during testing phase

# dataset/feature/channel
study = 1
input_type = 'raw' 
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 12 #[[12,12], [10,20]]  # load data from x seconds before each probe
subs = range(1,31)  # 1-30 for study 1, 301-330 for study 2
channels = range(32)


# cross validation
valMethod = None  # 'lnpo'(leave-n-participant-out); 'lopo'(leave-one-out)
if valMethod == 'lnpo':
    valRatio = 0.2 # validation size in percentage
else:
    valRatio = None

# hyperparameters 
batchSize = 120
nEpoch = 200  
trainBalanceOn = True  # &&& balancing not compatible with cupy
valBalanceOn = False
balanceMethod = 'oversample'  # ['oversample', 'undersample', 'smote']

dropout = 0.4  # 0 if off

# hyperparameters (tunable)
classWeight = 1.0  # weight for the second class, compared to 1 for the first class
norm = 'off'  # percentage scales the data when loading data. One of ['off', 'chan', 'freq', 'chanfreq', 'trial', 'signal']

F1 = 8 # number of temporal filters
D = 2 # number of spatial filter (for raw) or frequency filter (other inputs)
F2 = F1*D # number of pointwise filters

    
lr = 0.0001
reg_type = None # ['l1', 'l2', 'None']  kernel_regularizer
reg_val = 0



###   configuration   ###
if testOn:
    if study == 1:
        subs = range(1,6)
    else:
        subs = range(301,306)
    nEpoch = 10
    tasks = [tasks.copy()[0]]
    channels = range(2)

# output path
model_name = input('Name the model: ')
if not testOn:
    f_hist_main = os.path.join('.', 'history', 'EEGnet2018', str(randseed), model_name)
else:
    f_hist_main = os.path.join('.', 'history', 'temp_codetest', str(randseed), model_name)

# create path if not existing
if not os.path.exists(f_hist_main):
    os.makedirs(f_hist_main)
    print('Create output directory')
else:
    if testOn: (print('Will overwrite...'))
    else:     
        resp = input('Output directory EXISTS. Overwrite?[y/n]')
        if not resp.lower() == 'y': sys.exit()
print('Output directory: {}'.format(f_hist_main))

# print settings
df_pars = pd.DataFrame({'parameter': ['rand_seed', 'study', 'input_type', 'conditions', 'tasks',
                                      'channels', 'participants', 
                                      'validation', 'val_size',
                                      'batch_size', 'n_epoch', 'balance_train', 'balance_val', 'balance_method',
                                      'dropout', 'class_{}_weight'.format(conds[1]),
                                      'normalization', 'n_temporal_filter', 'n_s/f_filter', 'n_pointwise_filter',
                                      'learning_rate', 'regularizer', 'reg_val', 'window_length'],
                        'value':[randseed, study, input_type, conds, tasks,
                                 channels, subs, 
                                 valMethod,
                                 valRatio if not valMethod == 'lopo' else 'N/A',
                                 batchSize, nEpoch,
                                 1 if trainBalanceOn else 0,
                                 1 if valBalanceOn else 0,
                                 balanceMethod,
                                 dropout, classWeight, norm, F1, D, F2,  
                                 lr, reg_type,
                                 reg_val if not reg_type == None else 'N/A',
                                 win_lens]})
df_pars.to_csv(os.path.join(f_hist_main, 'settings.csv'), index = False)

print('############ current setting ############')
print(df_pars)
print('#########################################')
resp = input('Please check the setting. Continue?[y/n]')
if not resp.lower() == 'y': sys.exit()


# load one data to configure the dimensions of input
if study == 1:
    tpx, tpy, tps = ld1.load_dataset_n([subs[0]], tasks[0], [input_type], conds, 3, False, 'all', gpu2use)
else:
    tpx, tpy, tps = ld2.load_dataset_n([subs[0]], tasks[0], [input_type], conds, 3, False, 'all', gpu2use)
nClass = len(conds)
nPnt = tpx.shape[2]
if input_type == 'raw':
    nDim = tpx.shape[1]
else:
    nDim = [tpx.shape[1], tpx.shape[3]]

# print settings:

if input_type == 'raw':
    print('Input data shape: %i spatial points, %i time points' %(nDim,nPnt))
else:
    print('Input data shape: %i spatial points, %i time points, %i frequency points' % (nDim[1], nPnt, nDim[0]))
print('Classify between %i classes' %nClass)
resp = input('Continue?[y/n]')
if not resp.lower() == 'y': sys.exit()


# split participants
def split_subs_lnpo(subs, val_ratio = 0.2, randSeed = None):
    np.random.seed(randSeed)
    subs = list(subs)
    np.random.shuffle(subs)
    nSub = len(subs)

    nSplit = int(1/val_ratio)
    nTest_lw = int(np.floor(nSub/nSplit))
    nTest_hi = int(np.ceil(nSub/nSplit))

    if nTest_lw == nTest_hi:
        nTests = [nTest_lw]*nSplit
    else:
        nTests = [nTest_hi]*(nSub-nSplit*nTest_lw)
        nTests.extend([nTest_lw]*(nSplit - len(nTests)))

    subs_test = []
    subs_train = []
    for li in range(nSplit):
        if li == 0:
            subs_test.append(subs[:np.cumsum(nTests)[li]])
        else:
            subs_test.append(subs[np.cumsum(nTests)[li-1]:np.cumsum(nTests)[li]])
        subs_train.append(list(compress(subs, ~np.in1d(subs, subs_test[li]))))

    return subs_train, subs_test


def subset_data_subs(x,y,s,subs):
    x2 = x[np.in1d(s, subs),]
    y2 = y[np.in1d(s, subs)]

    return x2,y2


def list2array(a, padding = 0):
# turn list a (of vectors varying in length) in an array b
# padding with number specified with "padding" at the end of each row
    b = np.ones([len(a), len(max(a, key=lambda x: len(x)))])
    b = b*padding
    for i, j in enumerate(a):
        b[i][0:len(j)] = j

    return b


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


# save training/validation participants
if valMethod == 'lnpo':
    trainsubs, testsubs = split_subs_lnpo(subs, valRatio, randseed)
elif valMethod == 'lopo':
    trainsubs, testsubs = split_subs_lnpo(subs, 1/len(subs), randseed)

if valMethod is not None:
    np.savetxt(os.path.join(f_hist_main, 'subs_train.csv'), list2array(trainsubs), delimiter=',', fmt='%i')
    np.savetxt(os.path.join(f_hist_main, 'subs_test.csv'), list2array(testsubs), delimiter=',', fmt='%i')



###   modelling   ###
def modelling(x_train, y_train, x_test, y_test, lnpoi):

    # preprocess
    x_train = np.expand_dims(x_train, 3)
    yc_train = keras.utils.to_categorical(y_train, num_classes=nClass)
    if valMethod is not None:
        x_test = np.expand_dims(x_test, 3)
        yc_test = keras.utils.to_categorical(y_test, num_classes=nClass)

    # configure
    C = x_train.shape[1]
    T = x_train.shape[2]
    if input_type == 'raw':
        srate = 128

    if input_type == 'raw':
        f_hist = f_hist_main # a virtual channel to indicate 'all'
        
    if not os.path.exists(f_hist):
        os.makedirs(f_hist)
        print('Create output directory: {}'.format(f_hist))

    if reg_type == 'l1':
        reg = l1(reg_val)
    elif reg_type == 'l2':
        reg = l2(reg_val)
    else:
        reg = None

    #es_base = EarlyStopping(monitor='val_loss', mode='min', patience=8)
    #es_channel = EarlyStopping(monitor='val_loss', mode='min', patience=2)

    # model design
    keras.backend.clear_session()

    inputs = Input(shape = x_train.shape[1:], name = 'input')

    hidden_layers = Sequential(name='hidden')
    
    if True:
        hidden_layers.add(Conv2D(F1, (1, int(srate/2)), padding = 'same', 
                                 activation = 'relu', kernel_regularizer = reg,
                                 name = 'Block1_Conv2D'))
        hidden_layers.add(BatchNormalization(name = 'Block1_BatchNorm1'))
        
        hidden_layers.add(DepthwiseConv2D((C,1), padding = 'valid',
                                          activation = 'relu', kernel_regularizer = reg,
                                          depth_multiplier = D, kernel_constraint=max_norm(1),
                                          name = 'Block1_DepthwiseConv2D'))
        
        hidden_layers.add(BatchNormalization(name = 'Block1_BatchNorm2'))
        hidden_layers.add(Activation('elu', name = 'Block1_Activation'))
        hidden_layers.add(AveragePooling2D(pool_size=(1,4),name='Block1_AveragePool2D'))
        hidden_layers.add(Dropout(dropout, name = 'Block1_Dropout'))

        hidden_layers.add(SeparableConv2D(F2, int(srate/2/4), padding = 'same',
                                          activation = 'relu', kernel_regularizer = reg,
                                          name = 'Block2_SeparableConv2D'))
        
        hidden_layers.add(BatchNormalization(name = 'Block2_BatchNorm'))
        hidden_layers.add(Activation('elu', name = 'Block2_Activation'))
        hidden_layers.add(AveragePooling2D(pool_size=(1,8),name='Block2_AveragePool2D'))
        hidden_layers.add(Dropout(dropout, name = 'Block2_Dropout'))

    hidden_layers.add(Flatten())

    last_hidden = hidden_layers(inputs)
    outputs = Dense(nClass, activation='softmax', name='Classfier')(last_hidden)

    model = Model(inputs = inputs, outputs = outputs)
    if lnpoi == 0:
        hidden_layers.summary()
        plot_model(hidden_layers, show_shapes=True, to_file=os.path.join(f_hist_main, 'model_hidden.png'))
        plot_model(model, show_shapes=True, to_file=os.path.join(f_hist_main, 'model.png'))
        model.summary()

    # optimizer
    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',  
                  optimizer=opt,
                  metrics=['accuracy', keras.metrics.Precision()])

    #if saveModelOn: model.save(os.path.join(f_hist, 'model_init.h5'))

    weights = {0: 1, 1: classWeight}
    if valMethod is not None:
        hist = model.fit(x_train, yc_train, class_weight=weights,
                         validation_data=(x_test, yc_test),
                         batch_size=batchSize, epochs=nEpoch)
                         #callbacks=[es_base])
        preds = model.predict(x_test)
    else:
        hist = model.fit(x_train, yc_train, class_weight=weights,
                         batch_size=batchSize, epochs=nEpoch)
                         #callbacks=[es_base])
    if saveModelOn:
        f_save = os.path.join(f_hist, 'model_lnpo{}.h5'.format(lnpoi))
        model.save(f_save)

    # load model (test only)
    if testOn:
        del model
        print('Model deleted.')
        model = load_model(f_save)
        preds2check = model.predict(x_test)
        if np.array_equal(preds, preds2check):
            print('Correclty reload the model.')
        else:
            print('WARNING Model incorreclty save.')

    # save predictions (as inputs for the meta-learner)
    yhat_train = model.predict(x_train)    
    yhat_train = np.argmax(yhat_train, axis = -1)
    df_train = eval_performance(yhat_train, y_train, nClass)
    f_train = plot_roc(yhat_train, y_train)
    df_train.to_csv(os.path.join(f_hist, 'metrics_train_seed{}_valiation{}.csv'.format(randseed, lnpoi)))
    f_train.savefig(os.path.join(f_hist, 'roc_train_seed{}_validation{}.png'.format(randseed,lnpoi)),
                    dpi = 300, format = 'png')

    if valMethod is not None:
        yhat_test = model.predict(x_test)
        yhat_test = np.argmax(yhat_test, axis = -1)
        df_test = eval_performance(yhat_test, y_test, nClass)
        f_test = plot_roc(yhat_test, y_test)
        df_test.to_csv(os.path.join(f_hist, 'metrics_val_seed{}_valiation{}.csv'.format(randseed, lnpoi)))
        f_test.savefig(os.path.join(f_hist, 'roc_val_seed{}_validation{}.png'.format(randseed,lnpoi)),
                       dpi = 300, format = 'png')
        return yhat_train, yhat_test
    else:
        return yhat_train



def lnpocv(channels, norm=norm, winlen=win_lens, subs=subs):
        
    # load all datasets
    if study == 1:
        x_all, y_all, s_all = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [channels], gpu2use)
    else:
        x_all, y_all, s_all = ld2.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [channels], gpu2use)

    y_train_all = []  # to summarise performance through validation
    yhat_train_all = []
    if valMethod is not None:
        y_val_all = []
        yhat_val_all = []
    for lnpoi in tqdm(range(0,len(trainsubs)) if valMethod is not None else [0]):

        if valMethod is not None:
            # subset data/ train-test splits 
            subs2train = trainsubs[lnpoi].copy()
            subs2test = testsubs[lnpoi].copy()
        else:
            subs2train = subs

        x_train, y_train = subset_data_subs(x_all, y_all, s_all, subs2train)
        if valMethod is not None:
            x_test, y_test = subset_data_subs(x_all, y_all, s_all, subs2test)

        if gpu2use > -1:
            print('Cupy array conversion...')
            x_train = cp.asnumpy(x_train)
            y_train = cp.asnumpy(y_train)
            if valMethod is not None:
                x_test = cp.asnumpy(x_test)
                y_test = cp.asnumpy(y_test)
                        
        # balance class size
        if trainBalanceOn:
            if testOn:
                print('Original training class size: {}'.format(Counter(y_train)))  # test only

            if balanceMethod == 'undersample':
                ros = RandomUnderSampler(random_state=randseed)
            elif balanceMethod == 'oversample':
                ros = RandomOverSampler(random_state=randseed)
            else:
                ros = SMOTE(random_state=randseed)
                
            tpx = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
            x_res, y_res = ros.fit_resample(tpx, y_train)
            if testOn:
                print('Balanced training class size: {}'.format(Counter(y_res)))
            x_train = x_res.reshape((x_res.shape[0],)+x_train.shape[1:])
            y_train = y_res.copy()

        if valBalanceOn and valMethod is not None:
            if testOn:
                print('Original validation class size: {}'.format(Counter(y_test)))  # test only

            if balanceMethod == 'undersample':
                ros = RandomUnderSampler(random_state=randseed)
            else:
                ros = RandomOverSampler(random_state=randseed)
                
            tpx = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
            x_res, y_res = ros.fit_resample(tpx, y_test)
            if testOn:
                print('Balanced validation class size: {}'.format(Counter(y_res)))
            x_test = x_res.reshape((x_res.shape[0],)+x_test.shape[1:])
            y_test = y_res.copy()
            
        if valMethod is not None:
            yhat_train, yhat_test = modelling(x_train, y_train, x_test, y_test, lnpoi)
        else:
            x_test = None
            y_test = None
            yhat_train = modelling(x_train, y_train, x_test, y_test, lnpoi)

        if isinstance(y_train_all, list):
            y_train_all = y_train.copy()
            yhat_train_all = yhat_train.copy()
            if valMethod is not None:
                y_val_all = y_test.copy()
                yhat_val_all = yhat_test.copy()
        else:
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            yhat_train_all = np.concatenate((yhat_train_all, yhat_train), axis=0)
            if valMethod is not None:
                y_val_all = np.concatenate((y_val_all, y_test), axis=0)
                yhat_val_all = np.concatenate((yhat_val_all, yhat_test), axis=0)
        
    # overall assessment
    df_train_all = eval_performance(yhat_train_all, y_train_all, nClass)
    f_train_all = plot_roc(yhat_train_all, y_train_all)
    df_train_all.to_csv(os.path.join(f_hist_main, 'metrics_train.csv'))
    f_train_all.savefig(os.path.join(f_hist_main, 'roc_train.png'),
                        dpi = 300, format = 'png')
    if valMethod is not None:
        df_val_all = eval_performance(yhat_val_all, y_val_all, nClass)
        f_val_all = plot_roc(yhat_val_all, y_val_all)
        df_val_all.to_csv(os.path.join(f_hist_main, 'metrics_val.csv'))
        f_val_all.savefig(os.path.join(f_hist_main, 'roc_val.png'),
                          dpi = 300, format = 'png')

        
# main function
lnpocv(channels = channels)

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)
