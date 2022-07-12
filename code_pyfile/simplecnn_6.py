# the script is to train input model for the metalearner
# each input model is trained with one EEG input type of one channel
# inter-subject modelling
# including validation 


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
study = 1
input_type = 'raw' # one of ['raw', 'power', 'ispc', 'wst']
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 12 #[[12,12], [10,20]]  # load data from x seconds before each probe

#ci_start = input('Please type the first channel id: ')
#ci_end = input('Please type the end channel id (excl.): ')
#channels = range(int(ci_start),int(ci_end))
channels = range(32)
subs = range(1,31)  # 1-30 for study 1, 301-330 for study 2

# training with subset 
subsSelectOn = False
if subsSelectOn:
    #subs2rm = [1,4,9,12,14,16,17,19,24,25,28,30]  # selection based on class ratio
    subs2rm = [2,3,5,7,9,10,12,16,18,21,22,23,27,29,30]  # selection based on var_idv/var_group
    
# noise adaptatiion
noiseAdaptOn = False

# cross validation
valMethod = 'lnpo'  # 'lnpo'(leave-n-participant-out); 'lopo'(leave-one-out)
if valMethod == 'lnpo':
    valRatio = 0.2 # validation size in percentage

# hyperparameters 
batchSize = 120
nEpoch = 200  
trainBalanceOn = True  # &&& balancing not compatible with cupy
valBalanceOn = False
balanceMethod = 'oversample'  # ['oversample', 'undersample', 'smote']

batchNormOn = False
dropout = 0.4  # 0 if off

# hyperparameters (tunable)W
classWeight = 1  # sweight for the second class, compared to 1 for the first class
norm = 'chan'  # percentage scales the data when loading data. One of ['off', 'chan', 'freq', 'chanfreq', 'trial', 'signal']

neurons = [16,32,32]  # neurons of each hidden layer; [16,32,32] for final design
fc_neurons = [200,50]  # fully connected layer neuron numbers
    
lr = 0.0001
reg_type = None
reg_val = 0



###   configuration   ###
if not testOn:
    if subsSelectOn:
        subs = list(subs)
        for sub2rm in subs2rm:
            try: subs.remove(sub2rm)
            except: continue
else:  # simplify the loop to reduce time in test mode
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
    f_hist_main = os.path.join('.', 'history', 'simplecnn_6', str(randseed), model_name)
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
                                      'noise_adaptation', 'validation', 'val_size',
                                      'batch_size', 'n_epoch', 'balance_train', 'balance_val', 'balance_method',
                                      'batch_norm', 'dropout', 'class_{}_weight'.format(conds[1]),
                                      'normalization', 'neurons',
                                      'learning_rate', 'regularizer', 'reg_val', 'window_length'],
                        'value':[randseed, study, input_type, conds, tasks,
                                 channels, subs, 
                                 1 if noiseAdaptOn else 0,
                                 valMethod,
                                 valRatio if not valMethod == 'lopo' else 'N/A',
                                 batchSize, nEpoch,
                                 1 if trainBalanceOn else 0,
                                 1 if valBalanceOn else 0,
                                 balanceMethod,
                                 1 if batchNormOn else 0,
                                 dropout, classWeight, norm, neurons,
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


# save training/validation participants
if valMethod == 'lnpo':
    trainsubs, testsubs = split_subs_lnpo(subs, valRatio, randseed)
elif valMethod == 'lopo':
    trainsubs, testsubs = split_subs_lnpo(subs, 1/len(subs), randseed)
np.savetxt(os.path.join(f_hist_main, 'subs_train.csv'), list2array(trainsubs), delimiter=',', fmt='%i')
np.savetxt(os.path.join(f_hist_main, 'subs_test.csv'), list2array(testsubs), delimiter=',', fmt='%i')



###   modelling   ###
def modelling(x_train, y_train, x_test, y_test, lnpoi, channel):

    # preprocess       
    yc_train = keras.utils.to_categorical(y_train, num_classes=nClass)
    yc_test = keras.utils.to_categorical(y_test, num_classes=nClass)


    f_hist = os.path.join(f_hist_main, str(channel))
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
    
    if input_type == 'raw':
        for i, nNeuron in enumerate(neurons):
            for j in range(1):  # layer repeats
                hidden_layers.add(Conv1D(nNeuron, 5, padding = 'same',
                                         activation = 'relu', kernel_regularizer = reg,
                                         name = 'block{}_conv{}'.format(i,j)))
                if batchNormOn: hidden_layers.add(BatchNormalization())
            hidden_layers.add(MaxPooling1D(pool_size=2,name='block{}_pool'.format(i)))
            if dropout>0: hidden_layers.add(Dropout(dropout))
        
    else:
        for i, nNeuron in enumerate(neurons):
            hidden_layers.add(Conv2D(nNeuron, (3,5), padding = 'same',
                                     activation = 'relu', kernel_regularizer = reg,
                                     name = 'conv{}'.format(i)))
            if batchNormOn: hidden_layers.add(BatchNormalization())
            hidden_layers.add(MaxPooling2D(pool_size=(2,2),name='pool{}'.format(i)))                                 
            if dropout>0: hidden_layers.add(Dropout(dropout))

    hidden_layers.add(Flatten())

    for i, nNeuron in enumerate(fc_neurons):
        hidden_layers.add(Dense(nNeuron, activation='relu', kernel_regularizer=reg, name='fc{}'.format(i)))
        if dropout>0: hidden_layers.add(Dropout(dropout))

    last_hidden = hidden_layers(inputs)
    baseline_output = Dense(nClass, activation='softmax', name='baseline')(last_hidden)

    model = Model(inputs = inputs, outputs = baseline_output)
    if lnpoi == 0 and channel == channels[0]:
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
    hist = model.fit(x_train, yc_train, class_weight=weights,
                     validation_data=(x_test, yc_test),
                     batch_size=batchSize, epochs=nEpoch)
                     #callbacks=[es_base])
    preds = model.predict(x_test)

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
    yhat_test = model.predict(x_test)
    np.savetxt(os.path.join(f_hist, 'pred_train_lnpo{}.txt'.format(lnpoi)), yhat_train, delimiter = ',')
    np.savetxt(os.path.join(f_hist, 'pred_val_lnpo{}.txt'.format(lnpoi)), yhat_test, delimiter=',')
    np.savetxt(os.path.join(f_hist, 'y_train_lnpo{}.txt'.format(lnpoi)), y_train, delimiter=',')
    np.savetxt(os.path.join(f_hist, 'y_val_lnpo{}.txt'.format(lnpoi)), y_test, delimiter=',')
    
    # check if predictions are saved and can be loaded correctly (test only)
    if testOn:
        pred_train = np.genfromtxt(os.path.join(f_hist, 'pred_train_lnpo{}.txt'.format(lnpoi)), delimiter = ',')
        print('Training hats correctly saved.') if np.array_equal(pred_train, yhat_train) else print('Training hats saving ERROR!')
        pred_val = np.genfromtxt(os.path.join(f_hist, 'pred_val_lnpo{}.txt'.format(lnpoi)), delimiter=',')
        print('Validation hats correctly saved.') if np.array_equal(pred_val, yhat_test) else print('Validation hats saving ERROR!')

    if noiseAdaptOn:
        
        # ignore baseline loss in training
        BETA = 0
        # build confusion matrix (prediction,noisy_label)
        ybaseline_predict = model.predict(x_train, batch_size=batchSize)
        ybaseline_predict = np.argmax(ybaseline_predict, axis=-1)

        baseline_confusion = np.zeros((nClass, nClass))
        for n, p in zip(y_train, ybaseline_predict):
            baseline_confusion[p, n] += 1

        channel_weights = baseline_confusion.copy()
        channel_weights /= channel_weights.sum(axis=1, keepdims=True)
        # perm_bias_weights[prediction,noisy_label] = log(P(noisy_label|prediction))
        channel_weights = np.log(channel_weights + 1e-8)

        channeled_output = Channel(name='channel', weights=[channel_weights])(baseline_output)

        simple_model = Model(inputs=inputs, outputs=[channeled_output, baseline_output])
        if lnpoi == 0 and channel == channels[0]:
            simple_model.summary()
            plot_model(simple_model, show_shapes=True, to_file=os.path.join(f_hist_main, 'model_noiseAdapt.png'))

        simple_model.compile(loss='categorical_crossentropy',
                             loss_weights=[1. - BETA, BETA],
                             optimizer=opt,
                             metrics=['accuracy', keras.metrics.Precision()])

        unchannelled = False
        try:  # sometime cannot be channeled
            hist2 = simple_model.fit(x_train,
                                     [yc_train, yc_train],
                                     batch_size=batchSize,
                                     epochs=nEpoch,
                                     validation_data=(x_test, [yc_test, yc_test])
                                     #callbacks=[es_channel]
                                    )
        except:
            unchannelled = True
            print('Unable to train the channel layer for badly trained baseline model. CHANNEL {}'.format(channel))

        noiseAdapt_preds = simple_model.predict(x_test)

        if saveModelOn:
            f_save = os.path.join(f_hist, 'model_noiseAdapt_lnpo' + str(lnpoi) + '.h5')
            simple_model.save(f_save)

        # load model (test only)
        if testOn:
            del simple_model
            print('Model deleted.')
            simple_model = load_model(f_save, custom_objects={'Channel': Channel}, compile = False)
            preds2check = simple_model.predict(x_test)
            if np.array_equal(noiseAdapt_preds, preds2check):
                print('Correclty reload the model.')
            else:
                print('WARNING Model incorreclty save.')
            
        # get outcome (as inputs for stacking models). Note outcomes are BINARY
        yhat_train = simple_model.predict(x_train)[1]
        yhat_test = simple_model.predict(x_test)[1]
        np.savetxt(os.path.join(f_hist, 'pred_train_noiseAdapt_lnpo{}.txt'.format(lnpoi)), yhat_train, delimiter=',')
        np.savetxt(os.path.join(f_hist, 'pred_val_noiseAdapt_lnpo{}.txt'.format(lnpoi)), yhat_test, delimiter=',')

        # check if predictions are saved and can be loaded correctly (test only)
        if testOn:
            pred_train = np.genfromtxt(os.path.join(f_hist, 'pred_train_noiseAdapt_lnpo{}.txt'.format(lnpoi)), delimiter=',')
            print('Training hats correctly saved.') if np.array_equal(pred_train, yhat_train) else print(
                'Training hats saving ERROR!')
            pred_val = np.genfromtxt(os.path.join(f_hist, 'pred_val_noiseAdapt_lnpo{}.txt'.format(lnpoi)), delimiter=',')
            print('Validation hats correctly saved.') if np.array_equal(pred_val, yhat_test) else print(
                'Validation hats saving ERROR!')
        
    if noiseAdaptOn:
        return unchannelled




def lnpocv(channel, norm=norm, winlen=win_lens, subs=subs):
    # load all datasets
    if study == 1:
        x_all, y_all, s_all = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [[channel]], gpu2use)
    else:
        x_all, y_all, s_all = ld2.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlen, norm, [[channel]], gpu2use)

    if noiseAdaptOn: unchannelled = []
    
    for lnpoi in tqdm(range(0,len(trainsubs))):
        
        # subset data/ train-test splits 
        subs2train = trainsubs[lnpoi].copy()
        subs2test = testsubs[lnpoi].copy()

        x_train, y_train = subset_data_subs(x_all, y_all, s_all, subs2train)
        x_test, y_test = subset_data_subs(x_all, y_all, s_all, subs2test)

        if gpu2use > -1:
            print('Cupy array conversion...')
            x_train = cp.asnumpy(x_train)
            x_test = cp.asnumpy(x_test)
            y_train = cp.asnumpy(y_train)
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

        if valBalanceOn:
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

        # batch x (freq x) time 
        if input_type == 'raw':
            x_train = np.transpose(x_train, (0, 2, 1))
            x_test = np.transpose(x_test, (0, 2, 1))

        if noiseAdaptOn:
            if modelling(x_train, y_train, x_test, y_test, lnpoi, channel):
                unchannelled.append([channel,lnpoi])
                
        else:
            modelling(x_train, y_train, x_test, y_test, lnpoi, channel)
            
    if noiseAdaptOn: return unchannelled


        
# main function
if noiseAdaptOn: unchannelled_all = []

for channel in channels:
    if noiseAdaptOn:
        unchannelled_all.extend(lnpocv(channel = channel))
    else:
        lnpocv(channel = channel)

if noiseAdaptOn:
    unchannelled_all = np.array(unchannelled_all)
    np.savetxt(os.path.join(f_hist_main, 'unchannelled.csv'), unchannelled_all, delimiter=',', fmt='%i')

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)
