
testOn = True  # f_hist will be a temporary folder
# device number
gpu2use = 2

saveModelOn = True

if not saveModelOn:
    resp = input('SaveModel is OFF. Continue?[y/n]')
    if not resp == 'y':
        import sys
        sys.exit()


if gpu2use > -1:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu2use], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
else:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import activations
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import os
import code_pyfile.loadData1 as ld1
import code_pyfile.loadData2 as ld2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from itertools import compress
import time
from tqdm import tqdm
from collections import Counter

t0 = time.perf_counter()


###   settings   ###

# randomization
randseed = 36 # [36, 54, 87]  # 27 during testing phase

# data and feature type
study = 1
feats = ['wst'] # [['raw'], ['power', 'ispc'], ['wst']]
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability
#subs2rm = [7,10,20,27]
#subs2rm = [7,10,15,27]
#subs2rm = [2,3,6,7,10,15,18,20,21,23,26,27]

# hyperparameters (fixed)
batchSize = 500
nEpoch = 100  # 50
lnpo = 0.2  #0.2  # in percentage, also n-fold cv for 'idv_models'
balanceOn = True  # for training only
subsSelectOn = False


# hyperparameters (tuning)
#hpar = 'classweights'
#hvals = [1, 1.2, 1.4, 1.6, 1.8, 2]  # weight for the second class, compared to 1 for the first class

#hpar = 'normalization'
#hvals = ['off','dataset','trial', 'chan']  #hvals = ['off', 'chan', 'trial', 'signal']  # for raw
#hvals = ['off', 'chan', 'freq', 'chanfreq', 'trial', 'signal']  # for power/ispc/wst
#hvals = ['off', 'chan', 'scale', 'chanscale', 'trial', 'signal']  # for wst, scale == freq, anther saying

#hpar = 'neurons'
#hvals = [128, 64, 32, 16]
#hvals = [32, 16, 8]
#hvals = [#[1024, 1024], [1024, 512],
         #[512, 512], [512, 256], # [512, 128], [512, 64],
         #[256, 256], [256, 128], #[256, 64],
         #[128, 128], [128, 64], #, [128,32], [128,16],
         #[64, 64], [64, 32], #[64, 16],
         #[32, 32], [32, 16]]
#hvals = [[32, 32], [32, 16], [32, 8], [16, 16], [16,8], [8,8]]

#hvals = [[32, 64], [32, 64, 128], [32, 64, 128, 256],
#         [64, 128], [64, 128, 256],
#         [128, 256]]

#hpar = 'learningrate'
#hvals = [0.0005, 0.0001]

#hpar = 'leaveout_participant'
#hvals = range(1,31)

#hpar = 'idv_models'
#hvals = range(1,31)

#hpar = 'wst26'
#hvals = [0]

#hpar = 'reg_l2'
#hpar = 'reg_l1'
#hvals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

#hpar = 'window_length'
#hvals = [10,20,30,40,50,60,90,120]

hpar = 'channel'
hvals = range(32)
#hvals = range(30)


# output path
if not testOn:
    f_hist_main = os.path.join('.', 'history', 'simplecnn_5_' + ''.join(feats), hpar+'_'+str(randseed))
else:
    f_hist_main = os.path.join('.', 'history', 'temp_codetest', hpar+'_'+str(randseed))

# create path if not existing
if not os.path.exists(f_hist_main):
    os.makedirs(f_hist_main)
    print('Create output directory')


###   configuration   ###
if not testOn:
    if study == 1:
        subs = range(1,31)
    else:
        subs = range(301,331)

    if subsSelectOn:
        subs = list(subs)
        for sub2rm in subs2rm:
            try: subs.remove(sub2rm)
            except: continue

    if hpar == 'idv_models':
        subs = hvals
else:  # make sure enough particitpants for test
    if study == 1:
        subs = range(1,6)
    else:
        subs = range(301,306)
    nEpoch = 10
    tasks = [tasks.copy()[0]]



# load one data to configure the dimension property
if study == 1:
    tpx, tpy, tps = ld1.load_dataset_n([subs[0]], tasks[0], feats, conds, 3, False)
else:
    tpx, tpy, tps = ld2.load_dataset_n([subs[0]], tasks[0], feats, conds, 3, False)
nClass = len(conds)
nPnt = tpx.shape[2]
if feats[0] == 'raw':
    nDim = tpx.shape[1]
else:
    nDim = [tpx.shape[1], tpx.shape[3]]


# print settings:
if feats[0] == 'raw':
    print('Input data shape: %i spatial points, %i time points' %(nDim,nPnt))
else:
    print('Input data shape: %i spatial points, %i time points, %i frequency points' % (nDim[1], nPnt, nDim[0]))
print('Classify between %i classes' %nClass)



# split participants
def split_subs_lnpo(subs, lnpo = 0.2, randSeed = None):
    np.random.seed(randSeed)
    subs = list(subs)
    np.random.shuffle(subs)
    nSub = len(subs)

    nSplit = int(1/lnpo)
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

def ifelse(statement, a,b):
    if statement:
        return a
    else:
        return b

def list2array(a, padding = 0):
# turn list a (of vectors varying in length) in an array b
# padding with number specified with "padding" at the end of each row
    b = np.ones([len(a), len(max(a, key=lambda x: len(x)))])
    b = b*padding
    for i, j in enumerate(a):
        b[i][0:len(j)] = j

    return b


# train/test(val) splits by subs (lnpo)
if not hpar == 'idv_models':
    trainsubs, testsubs = split_subs_lnpo(subs, lnpo, randseed)
    np.savetxt(os.path.join(f_hist_main, 'subs_train.csv'), list2array(trainsubs), delimiter = ',', fmt='%i')
    np.savetxt(os.path.join(f_hist_main, 'subs_test.csv'), list2array(testsubs), delimiter = ',', fmt='%i')


###   modelling   ###
def modelling(hpar, hval, x_train, y_train, x_test, y_test, lnpoi):

    # preprocess
    yc_train = keras.utils.to_categorical(y_train, num_classes=nClass)
    yc_test = keras.utils.to_categorical(y_test, num_classes=nClass)

    f_hist = os.path.join(f_hist_main, str(hval))
    if not os.path.exists(f_hist):
        os.makedirs(f_hist)
        print('Create output directory')

    if hpar == 'neurons':
        neurons = hval
    else:
        #neurons = [64,128,256,512]  # M1
        neurons = [64, 128]  # M2
        #neurons = [64, 16]   # M3

    if hpar == 'reg_l2':
        reg = l2(hval)
    elif hpar == 'reg_l1':
        reg = l1(hval)
    else:
        reg = None

    # model design
    keras.backend.clear_session()

    inputs = Input(shape = x_train.shape[1:], name = 'input')
    x = Conv2D(neurons[0], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block1_conv1')(inputs)
    #x = Conv2D(neurons[0], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2,2), name = 'block1_pool')(x)

    x = Conv2D(neurons[1], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block2_conv1')(x)
    #x = Conv2D(neurons[1], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2,2), name = 'block2_pool')(x)

    if False:
        x = Conv2D(neurons[2], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block3_conv1')(x)
        x = Conv2D(neurons[2], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block3_conv2')(x)
        x = Conv2D(neurons[2], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block3_conv3')(x)
        x = MaxPooling2D(pool_size=(2,2), name = 'block3_pool')(x)

        x = Conv2D(neurons[3], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block4_conv1')(x)
        x = Conv2D(neurons[3], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block4_conv2')(x)
        x = Conv2D(neurons[3], (3,3), activation = 'relu', kernel_regularizer = reg, name = 'block4_conv3')(x)
        x = MaxPooling2D(pool_size=(2,2), name = 'block4_pool')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=reg, name='fc1')(x)
    x = Dense(1000, activation = 'relu', kernel_regularizer=reg, name = 'fc2')(x)
    outputs = Dense(2, activation='softmax', name = 'predictions')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.summary()

    if saveModelOn:
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(f_hist, 'model_init.json'), "w") as json_file:
            json_file.write(model_json)

    if hpar == 'learningrate':
        lr = hval
    else:
        lr = 0.001

    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',  # 'binary_crossentropy'
                  optimizer=opt,
                  metrics=['accuracy'])

    if hpar == 'classweights':
        weights = {0: 1, 1: hval}
    else:
        weights = {0: 1, 1: 1}


    hist = model.fit(x_train, yc_train, class_weight=weights,
                     validation_data=(x_test, yc_test),
                     batch_size=batchSize, epochs=nEpoch)
    preds = model.predict(x_test)
    y_hat = np.argmax(preds, axis = -1)
    report = classification_report(y_test, y_hat, [0, 1], output_dict=True)

    df = pd.DataFrame(hist.history)
    df['specificity'] = report['0']['recall']
    df['sensitivity'] = report['1']['recall']
    if report['1']['recall'] == 0 :  # ill-defined precision - no positive predictions makes precision =0
        df['precision'] = -1
    else:
        df['precision'] = report['1']['precision']

    df['nTrain'] = x_train.shape[0]
    df['nVal'] = x_test.shape[0]
    df['nVal_pos'] = report['1']['support']
    df['randseed'] = randseed

    if saveModelOn: model.save_weights(os.path.join(f_hist, 'weights_lnpo' + str(lnpoi) + '.h5'))
    df.to_csv(os.path.join(f_hist, 'lnpo' + str(lnpoi) + '.csv'))



def lnpocv(norm, winlen=20, sidx='all', subs=subs):
    # load all datasets
    if study == 1:
        x_all, y_all, s_all = ld1.load_dataset_n_tasks(subs, tasks, feats, conds, winlen, norm, sidx)
    else:
        x_all, y_all, s_all = ld2.load_dataset_n_tasks(subs, tasks, feats, conds, winlen, norm, sidx)


    for lop in ifelse(hpar == 'leaveout_participant', hvals, [0]):

        if hpar == 'idv_models':

            if len(Counter(y_all).keys()) < 2: # if either class is missing
                continue

            kf = KFold(n_splits = int(1/lnpo), shuffle = True, random_state = randseed)
            trainlist = []
            testlist = []
            for train_index, test_index in kf.split(x_all):
                trainlist.append(train_index)
                testlist.append(test_index)

        for lnpoi in tqdm(range(int(1/lnpo))):

            # subset data/ train-test splits
            if not hpar == 'idv_models':
                subs2train = trainsubs[lnpoi].copy()
                subs2test = testsubs[lnpoi].copy()

                if hpar == 'leaveout_participant':
                    if np.isin(lop, subs2train):
                        subs2train.remove(lop)
                    else:
                        subs2test.remove(lop)

                x_train, y_train = subset_data_subs(x_all, y_all, s_all, subs2train)
                x_test, y_test = subset_data_subs(x_all, y_all, s_all, subs2test)

            else:
                x_train, y_train = x_all[trainlist[lnpoi]], y_all[trainlist[lnpoi]]
                x_test, y_test = x_all[testlist[lnpoi]], y_all[testlist[lnpoi]]


            if balanceOn:
                #Counter(y_train)  # test only
                ros = RandomOverSampler(random_state=randseed)
                tpx = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
                x_res, y_res = ros.fit_resample(tpx, y_train)
                #Counter(y_res)
                x_train = x_res.reshape((x_res.shape[0],)+x_train.shape[1:])
                y_train = y_res.copy()


            # batch x freq x time x chan
            if feats[0] == 'raw':
                x_train = np.transpose(x_train, (0, 2, 1))
                x_test = np.transpose(x_test, (0, 2, 1))


            if hpar == 'normalization':
                modelling(hpar, norm, x_train, y_train, x_test, y_test, lnpoi)
            elif hpar == 'leaveout_participant':
                modelling(hpar, lop, x_train, y_train, x_test, y_test, lnpoi)
            elif hpar == 'idv_models':
                modelling(hpar, subs[0], x_train, y_train, x_test, y_test, lnpoi)
            elif hpar == 'window_length':
                modelling(hpar, winlen, x_train, y_train, x_test, y_test, lnpoi)
            elif hpar == 'channel':
                modelling(hpar, sidx[0][0], x_train, y_train, x_test, y_test, lnpoi)
            else:
                for hi in range(len(hvals)):
                    modelling(hpar, hvals[hi], x_train, y_train, x_test, y_test, lnpoi)


if hpar == 'normalization':
    for hval in hvals:
        lnpocv(hval)
else:
    norm = 'off'

    if hpar == 'window_length':
        for hval in hvals:
            lnpocv(norm, winlen=hval)
    elif hpar == 'channel':
        for hval in hvals:
            lnpocv(norm, sidx=[[hval]])
    elif hpar == 'idv_models':
        for hval in hvals:
            lnpocv(norm, subs = [hval])
    else:
        lnpocv(norm)

t2 = time.perf_counter()

#print("Time loading data [hr]: ", (t1-t0)/3600) # CPU seconds elapsed (floating point)
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)
