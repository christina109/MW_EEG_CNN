# visualize the features of the stacking model

# device number
gpu2use = input('Please specify the gpu to use: ')
gpu2use = int(gpu2use)
#gpu2use = 3  # -1 if cpu


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
from code_pyfile.channel import Channel

meta_model = input('Specify the meta model: ')


validations = [0]
seed = 36

    
for vali in validations:
    model2load = os.path.join('history', 'metalearner_full', meta_model, 'metalearn_seed{}_validation{}.h5'.format(seed, vali))
    stacking = load_model(model2load, custom_objects = {'Channel': Channel})

    fc1 = stacking.layers[1].layers[0]
    fc2 = stacking.layers[1].layers[2]
    fin = stacking.layers[-1]

    w1 = fc1.get_weights()[0]
    w2 = fc2.get_weights()[0]
    wf = fin.get_weights()[0]

    w = np.matmul(w1,w2)
    w = np.matmul(w, wf)

    if vali == 0:
        w_all = w.copy()
    else:
        w_all = w_all + w

    if len(validations) == 5:
        ax = plt.subplot(2,3,vali+1)
        ax.matshow(np.reshape(w, (32,23)))

if len(validations)==5:
    ax = plt.subplot(2,3,6)
    ax.matshow(np.reshape(w_all/5, (32, 23)))
    plt.show()


w_mean = w_all/len(validations)
for xi in range(w_mean.shape[1]):
    ax = plt.subplot(1,2, xi+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.matshow(np.reshape(w_mean[:,xi], (8,8)))
plt.show()

np.savetxt('w_nn_wst_omni.csv', w_mean, delimiter = ',')


# load input model

input_type = 'wst_normScale_cw1.4'
model_id = 18 
seed = 36
vali = 0
model2load = os.path.join('history', 'simplecnn_6_omni', str(seed), input_type, str(model_id), 'model_lnpo{}.h5'.format(vali))
model = load_model(model2load)
for layer in model.layers[1].layers:
    if 'conv0' in layer.name:
        conv1 = layer
    if 'conv1' in layer.name:
        conv2 = layer
    if 'conv2' in layer.name:
        conv3 = layer
        break
    

if False:
    filters, biases = conv1.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters-f_min)/(f_max-f_min)

    ix, nFilter = 1, filters.shape[3]
    for i in range(nFilter):
        f = filters[:,:,0,i ]
        ax = plt.subplot(4,4,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f, cmap = 'gray')
        ix += 1
    plt.show()


# trial in manuscript (sub = 1, ti = 0)
dat,_ = ld1.load_dataset(2, 'sart', 'wst', ['mw'], 1, norm = 'scale', sidx = [model_id],gpu2use = -1, sessions = [1,2])
dat=dat[1,:,:,:]
dat = np.expand_dims(dat, axis = 0)

m1 = Model(inputs=conv1.input, outputs=conv1.output)
m2 = Model(inputs=conv1.input, outputs=conv2.output)
m3 = Model(inputs=conv1.input, outputs=conv3.output)


feature_maps = m1.predict(dat)
nMap = feature_maps.shape[3]
for fi in range(min(16,nMap)):
    ax = plt.subplot(4,4,fi+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_maps[0,:,:,fi])
plt.show()


feature_maps = m2.predict(dat)
nMap = feature_maps.shape[3]
for fi in range(min(16,nMap)):
    ax = plt.subplot(4,4,fi+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_maps[0,:,:,fi])
plt.show()

feature_maps = m3.predict(dat)
nMap = feature_maps.shape[3]
for fi in range(min(16,nMap)):
    ax = plt.subplot(4,4,fi+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_maps[0,:,:,fi])
plt.show()

