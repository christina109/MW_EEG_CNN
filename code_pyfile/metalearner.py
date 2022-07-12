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

#channels = [12,31,30,8,13,25,9,21,4,1,2,3,22,18,26,17,28,5]
#channels = range(120)

stacking_model = input('Stacking model type? ')
noiseAdaptOn = False  # active in 'nn' only
dropout = 0.2
lr = 0.0005
nEpoch_base = 50
nEpoch_adapt = 50
classWeight = 1

if True:
    f_model = input('Please specify the input model ')
    if f_model[:4] == 'ispc':
        channels = range(120)
    else:
        channels = range(32)
    
    input_models = {f_model: {'channel': channels}}
    
if False:
    input_models= {'raw_normChan_weight1.6':{'channel': range(32)},
                   'power_normChanfreq_weight1.2':{'channel': range(32)},
                   'ispc_normChanfreq_weight1.6':{'channel': range(120)},
                   'wst_normScale_weight1.4':{'channel': range(32)}}
if False:
    input_models= {#'raw_normChan': {'channel': range(32)},
                   #'power_normChanfreq':{'channel': range(32)},
                   #'ispc_normChanfreq':{'channel': range(120)},
                   'wst_normScale':{'channel': range(32)}}

neural_network = 'simplecnn_6'

model_type = ''  # ['', 'noiseAdapt_'] - '' for default CNN
seeds = [36]
val_indices = range(5)
valMethod = 'lnpo'  #['None', 'lnpo']


# output directory
f_save = input('Name the model: ')
p_save = os.path.join('history', 'metalearner', f_save)
if not os.path.exists(p_save):
    os.makedirs(p_save)
    print('Create output directory')
else:
    if testOn: (print('Will overwrite...'))
    else:     
        resp = input('Output directory EXISTS. Overwrite?[y/n]')
        if not resp.lower() == 'y': sys.exit()
print('Output directory: {}'.format(p_save))
    

# print settings
df_ml = pd.DataFrame({'parameter':['input_models','neural_network', 'model_type',
                                   'seeds', 'val_indices', 'val_method', 'stacking_model',
                                   'stacking_noiseAdapt', 'dropout', 'learning_rate',
                                   'epoch_size_base', 'epoch_size_adapt', 'class_weight'],
                      'value':[input_models, neural_network, model_type,
                               seeds, val_indices, valMethod, stacking_model,
                               1 if noiseAdaptOn else 0, dropout, lr,
                               nEpoch_base, nEpoch_adapt, classWeight]})
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
y_train_all = []  # to summarise performance through validation
yhat_train_all = []
if valMethod is not None:
    y_val_all = []
    yhat_val_all = []
input_list = []
for seed in tqdm(seeds):
    for vali in tqdm(val_indices):

        X_train = []
        y_train = []
        if valMethod is not None:
            X_val = []
            y_val = []
        
        nClass = 0

        for input_type in tqdm(input_models):

            print('Loading data...')
            
            p_input_model = os.path.join('.', 'history', neural_network, str(seed), input_type)
            df_pars = pd.read_csv(os.path.join(p_input_model, 'settings.csv'), index_col = 0)

            if nClass == 0:
                nClass = len(eval(df_pars.value['conditions']))
            else:
                if not nClass == len(eval(df_pars.value['conditions'])):
                    resp = input('UNMATCHED CLASSIFICATION between input models. Continue?[y/n]')
                    if not resp.lower() == 'y': sys.exit()

                                     
            channels = input_models[input_type]['channel']
                     
            for channel in channels:

                if vali == 0 and seed == seeds[0]:
                    input_list.append('{}_{}'.format(input_type, channel))
                    input_list.append('{}_{}'.format(input_type, channel))

                f_train_x = os.path.join(p_input_model, str(channel),
                                       'pred_train_{}lnpo{}.txt'.format(model_type, vali))

                f_train_y = os.path.join(p_input_model, str(channel),
                                       'y_train_lnpo{}.txt'.format(vali))
                
                pred_train = np.genfromtxt(f_train_x, delimiter = ',')
                y_train_tp = np.genfromtxt(f_train_y, delimiter = ',')

                
                if valMethod is not None:
                    f_val_x = os.path.join(p_input_model, str(channel),
                                           'pred_val_{}lnpo{}.txt'.format(model_type, vali))
                    f_val_y = os.path.join(p_input_model, str(channel),
                                           'y_val_lnpo{}.txt'.format(vali))
                    pred_val = np.genfromtxt(f_val_x, delimiter = ',')
                    y_val_tp = np.genfromtxt(f_val_y, delimiter = ',')
                        

                if isinstance(X_train, list):
                    X_train = pred_train.copy()
                    if valMethod is not None:
                        X_val = pred_val.copy()
                else:
                    X_train = np.concatenate((X_train, pred_train), axis = 1)
                    if valMethod is not None:
                        X_val = np.concatenate((X_val, pred_val), axis = 1)

                if isinstance(y_train, list):
                    y_train = y_train_tp.copy()

                    if not y_train.shape[0] == X_train.shape[0]:
                        print('UNMATCHED X/Y. Abort session.')
                        sys.exit()
                    if valMethod is not None:
                        y_val = y_val_tp.copy()
                        if not y_val.shape[0] == X_val.shape[0]:
                            print('UNMATCHED X/Y. Abort session.')
                            sys.exit()
                else:
                    if not np.array_equal(y_train, y_train_tp):
                         print('UNMATCHED Y between input models. Abort session.')
                         sys.exit()
                    if valMethod is not None:
                         if not np.array_equal(y_val, y_val_tp):
                             print('UNMATCHED Y between input models. Abort session.')
                             sys.exit()
                     
        ### meta-learning  ###
        print('Modelling...')
                                     
        if stacking_model == 'lr':
                                     
            model = LogisticRegression().fit(X_train, y_train)
            yhat_train = model.predict(X_train)
            if valMethod is not None:
                yhat_val = model.predict(X_val)

            f_output = os.path.join(p_save, 'metalearn_seed{}_validation{}.pkl'.format(seed, vali))
            with open(f_output, 'wb') as of: pickle.dump(model, of)

            if testOn and valMethod is not None:                        
                del model
                print('Model deleted.')
                with open(f_output, 'rb') as lf: model = pickle.load(lf)
                yhat_tp = model.predict(X_train)
                if np.array_equal(yhat_tp, yhat_train):
                    print('Reload model correctly.')
                else:
                    resp = input('Model incorrectly save. Continue?[y/n]')
                    if not resp.lower() == 'y': sys.exit()

        elif stacking_model == 'nn':
            yc_train = keras.utils.to_categorical(y_train, num_classes=nClass)
            if valMethod is not None:
                yc_val = keras.utils.to_categorical(y_val, num_classes=nClass)
    
            keras.backend.clear_session()
            inputs = Input(shape = X_train.shape[1:], name = 'input')
            hidden_layers = Sequential(name = 'hidden')
            hidden_layers.add(Dense(100, activation = 'relu', name = 'fc1'))
            if dropout>0: hidden_layers.add(Dropout(dropout))
            hidden_layers.add(Dense(20, activation = 'relu', name = 'fc2'))
            if dropout>0: hidden_layers.add(Dropout(dropout))
            last_hidden = hidden_layers(inputs)
            baseline_output = Dense(nClass, activation = 'softmax', name = 'baseline')(last_hidden)
            model = Model(inputs = inputs, outputs = baseline_output)

            if seed == seeds[0] and vali == val_indices[0]:
                hidden_layers.summary()
                plot_model(hidden_layers, show_shapes=True, to_file = os.path.join(p_save, 'model_hidden.png'))
                plot_model(model, show_shapes=True, to_file = os.path.join(p_save, 'model.png'))
                model.summary()

            opt = keras.optimizers.Adam(learning_rate = lr)
            model.compile(loss='categorical_crossentropy',
                          optimizer = opt,
                          metrics = ['accuracy', keras.metrics.Precision()])

            cw = {0: 1, 1: classWeight}
            if valMethod is not None:
                hist = model.fit(X_train, yc_train, class_weight = cw,
                                 validation_data = (X_val, yc_val),
                                 batch_size = 200, epochs = nEpoch_base)
            else:
                hist = model.fit(X_train, yc_train, class_weight = cw,
                                 batch_size = 200, epochs = nEpoch_base)

            if not noiseAdaptOn:
                f_save = os.path.join(p_save, 'metalearn_seed{}_validation{}.h5'.format(seed, vali))
                model.save(f_save)
                yhat_train = model.predict(X_train)
                if valMethod is not None:
                    yhat_val = model.predict(X_val)

                if testOn and valMethod is not None:
                    del model
                    print('Model deleted.')
                    model = load_model(f_save)
                    preds2check = model.predict(X_train)
                    if np.array_equal(yhat_train, preds2check):
                        print('Correclty reload the model.')
                    else:
                        print('WARNING Model incorreclty save.')

            if noiseAdaptOn:
                BETA = 0
                ybaseline_predict = model.predict(X_train)
                ybaseline_predict = np.argmax(ybaseline_predict, axis = -1)
                baseline_confusion = np.zeros((nClass, nClass))
                y_train_int = np.argmax(yc_train, axis = -1)
                for n, p in zip(y_train_int, ybaseline_predict):
                    baseline_confusion[p, n] += 1

                channel_weights = baseline_confusion.copy()
                channel_weights/=channel_weights.sum(axis=1, keepdims=True)
                channel_weights = np.log(channel_weights + 1e-8)

                channeled_output = Channel(name='channel', weights=[channel_weights])(baseline_output)

                simple_model = Model(inputs=inputs, outputs=[channeled_output, baseline_output])
                if seed == seeds[0] and vali == val_indices[0]:
                    simple_model.summary()
                    plot_model(simple_model, show_shapes=True, to_file=os.path.join(p_save, 'model_noiseAdapt.png'))

                simple_model.compile(loss='categorical_crossentropy',
                                     loss_weights=[1. - BETA, BETA],
                                     optimizer=opt,
                                     metrics=['accuracy', keras.metrics.Precision()])
                try:  # sometime cannot be channeled
                    if valMethod is not None:
                        hist2 = simple_model.fit(X_train,
                                                 [yc_train, yc_train],
                                                 class_weight = cw, 
                                                 batch_size=200,
                                                 epochs=nEpoch_adapt,
                                                 validation_data=(X_val, [yc_val, yc_val])
                                                 #callbacks=[es_channel]
                                                )
                    else:
                        hist2 = simple_model.fit(X_train,
                                                 [yc_train, yc_train],
                                                 class_weight = cw, 
                                                 batch_size=200,
                                                 epochs=nEpoch_adapt,
                                                 #callbacks=[es_channel]
                                                )
                except:
                    print('Unable to channel')

                f_save = os.path.join(p_save, 'metalearn_seed{}_validation{}.h5'.format(seed, vali))
                simple_model.save(f_save)

                yhat_train = simple_model.predict(X_train)[1]
                if valMethod is not None:
                    yhat_val = simple_model.predict(X_val)[1]      

                # load model (test only)
                if testOn and valMethod is not None:
                    del simple_model
                    print('Model deleted.')
                    simple_model = load_model(f_save, custom_objects={'Channel': Channel}, compile = False)
                    preds2check = simple_model.predict(X_val)[1]
                    if np.array_equal(yhat_val, preds2check):
                        print('Correclty reload the model.')
                    else:
                        print('WARNING Model incorreclty save.')
     
            yhat_train = np.argmax(yhat_train, axis = -1)
            if valMethod is not None:
                yhat_val = np.argmax(yhat_val, axis = -1)
            # end of nn 

        
        
        # eval performance
        df_train = eval_performance(yhat_train, y_train, nClass)
        f_train = plot_roc(yhat_train, y_train)
        df_train.to_csv(os.path.join(p_save, 'metrics_train_seed{}_valiation{}.csv'.format(seed, vali)))
        f_train.savefig(os.path.join(p_save, 'roc_train_seed{}_validation{}.png'.format(seed,vali)),
                        dpi = 300, format = 'png')

        if valMethod is not None:
            df_val = eval_performance(yhat_val, y_val, nClass)
            f_val = plot_roc(yhat_val, y_val)
            df_val.to_csv(os.path.join(p_save, 'metrics_val_seed{}_valiation{}.csv'.format(seed, vali)))
            f_val.savefig(os.path.join(p_save, 'roc_val_seed{}_validation{}.png'.format(seed,vali)),
                          dpi = 300, format = 'png')

        # data for overall assessment
        if isinstance(y_train_all, list):
            y_train_all = y_train.copy()
            yhat_train_all = yhat_train.copy()
            if valMethod is not None:
                y_val_all = y_val.copy()
                yhat_val_all = yhat_val.copy()
        else:
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            yhat_train_all = np.concatenate((yhat_train_all, yhat_train), axis=0)
            if valMethod is not None:
                y_val_all = np.concatenate((y_val_all, y_val), axis=0)
                yhat_val_all = np.concatenate((yhat_val_all, yhat_val), axis=0)

pd.DataFrame(input_list).to_csv(os.path.join(p_save, 'input_list.csv'))

# overall assessment
df_train_all = eval_performance(yhat_train_all, y_train_all, nClass)
f_train_all = plot_roc(yhat_train_all, y_train_all)
df_train_all.to_csv(os.path.join(p_save, 'metrics_train.csv'))
f_train_all.savefig(os.path.join(p_save, 'roc_train.png'),
                    dpi = 300, format = 'png')
if valMethod is not None:
    df_val_all = eval_performance(yhat_val_all, y_val_all, nClass)
    f_val_all = plot_roc(yhat_val_all, y_val_all)
    df_val_all.to_csv(os.path.join(p_save, 'metrics_val.csv'))
    f_val_all.savefig(os.path.join(p_save, 'roc_val.png'),
                      dpi = 300, format = 'png')

t2 = time.perf_counter()
print("Time elapsed [min]: ", (t2-t0)/60) # CPU seconds elapsed (floating point)


