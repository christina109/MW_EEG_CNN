# same as the first stacking ensemble but computing the outcome of each model first
# device number
gpu2use = 2

if gpu2use > -1:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu2use], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
else:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model, Model, model_from_json
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, Concatenate
import keras
import numpy as np
import pandas as pd
import os
import code_pyfile.loadData1 as ld1
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from datetime import datetime
import pickle
from tqdm import tqdm
import time

t0 = time.perf_counter()

features = {'raw': {'channel': [5,22,25]},
            'wst': {'channel': [31,8,22]},
            'power': {'channel': [1,16,11]},
            'ispc': {'channel': [77,103,53]}}
nn = 'simplecnn_5'
#training = 0.5  # in percentage - furthur spliting the origial validation sample (stack_model_training vs. stack_model_validation)

seeds = [36, 54, 87]
lnpo = range(5)

winlen = 20
batchSize = 200
nEpoch = 50  # 50

# set/make output directory
p_save = os.path.join('history', 'stacking_ensemble', datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S'))
if not os.path.exists(p_save):
    os.makedirs(p_save)


df = pd.DataFrame(features)
df = df.transpose()
df = df.stack()

df.to_csv(os.path.join(p_save, 'features.csv'))


def ifelse(statement,a,b):
    if statement:
        return a
    else:
        return b



def config_normalization(feature):
    if feature == 'power':
        norm = 'off'
    elif feature == 'ispc':
        norm = 'off'
    elif feature == 'wst':
        norm = 'off'
    else:
        norm = 'off'

    return norm



def split_dataset(x,y,training,seed):

    np.random.seed(seed)
    n = x.shape[0]
    idx = list(range(n))
    np.random.shuffle(idx)
    idx_train = idx[:round(n*training)]
    idx_val = idx[round(n*training):]

    x_train = x[idx_train,]
    y_train = y[idx_train]

    if not training == 1:
        x_val = x[idx_val,]
        y_val = y[idx_val]
    else:
        x_val = x_train.copy()
        y_val = y_train.copy()

    return x_train, y_train, x_val, y_val


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



def balance_dataset(x_list,y,seed):

    singleList = not isinstance(x_list, list)
    if singleList:
       x_list = [x_list]


    x_b = []
    for x in x_list:
        #Counter(y)  # test only
        ros = RandomOverSampler(random_state=seed)
        tpx = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        x_res, y_res = ros.fit_resample(tpx, y)
        #Counter(y_res)
        x_b.append(x_res.reshape((x_res.shape[0],) + x.shape[1:]))
        y_b = y_res.copy()

    if singleList:
        x_b = x_b[0]

    return x_b, y_b



def remove_zeros(x):
    x = list(x)
    if 0 in x:
        x.remove(0)

    return x



# load models and data
def load_models_and_data(seed, lnpoi):

    all_x_train = list()
    all_x_val = list()
    all_y_train = list()
    all_y_val = list()

    for feat_hpar in df.index:
        feat = feat_hpar[0]
        hpar = feat_hpar[1]
        hvals = df[feat_hpar]

        norm = config_normalization(feat)

        for hval in hvals:

            p_feat = os.path.join('history', nn + '_' + feat, hpar + '_' + str(seed))
            subs_train = pd.read_csv(os.path.join(p_feat,'subs_train.csv'), header = None).iloc[lnpoi,]
            subs_val = pd.read_csv(os.path.join(p_feat, 'subs_test.csv'), header=None).iloc[lnpoi,]

            subs_train = remove_zeros(subs_train)
            subs_val = remove_zeros(subs_val)

            # load data
            if hpar == 'channel':
                x_train, y_train, _ = ld1.load_dataset_n_tasks(subs=subs_train, tasks=['sart', 'vs'], feats=[feat], conds=['ot', 'mw'],
                                                               winlen=winlen, norm=norm, sidx=[[hval]])
                x_val, y_val, _ = ld1.load_dataset_n_tasks(subs=subs_val, tasks=['sart', 'vs'], feats=[feat], conds=['ot', 'mw'],
                                                           winlen=winlen, norm=norm, sidx=[[hval]])
            else:
                x_train, y_train, _ = ld1.load_dataset_n_tasks(subs=subs_train, tasks=['sart', 'vs'], feats=[feat], conds=['ot', 'mw'],
                                                               winlen=winlen, norm=norm)
                x_val, y_val, _ = ld1.load_dataset_n_tasks(subs=subs_val, tasks=['sart', 'vs'], feats=[feat], conds=['ot', 'mw'],
                                                           winlen=winlen, norm=norm)

            if feat == 'raw':
                x_train = np.transpose(x_train, (0, 2, 1))
                x_val = np.transpose(x_val, (0, 2, 1))

            # load model
            p_model = os.path.join('history', nn + '_' + feat, hpar + '_' + str(seed), str(hval))
            with open(os.path.join(p_model, 'model_init.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(p_model, 'weights_lnpo' + str(lnpoi) + '.h5'))
            print('>loaded %s' % p_model)

            # get outcome
            yhat_train = model.predict(x_train)
            yhat_val = model.predict(x_val)

            # outcome from each model concatenated as the new input
            if all_x_train == []:
                all_x_train = yhat_train.copy()
                all_x_val = yhat_val.copy()
            else:
                all_x_train = np.concatenate((all_x_train, yhat_train), axis = 1)
                all_x_val = np.concatenate((all_x_val, yhat_val), axis=1)

            # only one 'y' is required
            if all_y_train == []:
                all_y_train = y_train.copy()
                all_y_val= y_val.copy()
            else:
                if not np.array_equal(all_y_train, y_train) or not np.array_equal(all_y_val, y_val):
                    import sys
                    sys.exit('Inputs not from the same trials')

    return all_x_train, all_y_train, all_x_val, all_y_val



# define stacked model from multiple member input models
def define_stacked_model():

    inputs = Input(shape = (sum(len(df[x]) for x in df.index)*2), name='input')
    hidden = Dense(12, activation='relu')(inputs)
    output = Dense(2, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def predict_stacked_model(model, x, y):
    yhat = model.predict(x)  # after balance
    yhat = np.argmax(yhat, axis=1)
    res = classification_report(y, yhat, [0, 1], output_dict=True)

    return res



def predict_logistic_model(model, x, y):
    yhat = model.predict(x)
    res = classification_report(y, yhat, [0, 1], output_dict=True)
    return res



def add_to_report(m_name, seed, lnpoi, dataset, res, report):

    specificity = res['0']['recall']
    sensitivity = res['1']['recall']
    if res['1']['recall'] == 0: # adjust an ill-defined precision
        precision = -1
    else:
        precision = res['1']['precision']
    accuracy = res['accuracy']

    report = report.append({'model':m_name,
                            'seed': seed,
                            'lnpoi': lnpoi,
                            'dataset': dataset,
                            'specificity': specificity,
                            'sensitivity': sensitivity,
                            'precision': precision,
                            'accuracy': accuracy},
                            ignore_index = True)

    return report



report = pd.DataFrame(columns=['model','seed','lnpoi','dataset','specificity','sensitivity','precision','accuracy'])
# when the seed is specified, sub_list.csv should be the same across feature/hpar
for seed in tqdm(seeds):

    for lnpoi in tqdm(lnpo):

        keras.backend.clear_session()

        # load data and models
        x_train, y_train, x_val, y_val = load_models_and_data(seed, lnpoi)

        # balance training datasets
        x_train_b, y_train_b = balance_dataset(x_train, y_train, seed)

        ### stacked by Neural Network  ###
        # define ensemble model
        stacked_model = define_stacked_model()
        # plot graph of ensemble
        plot_model(stacked_model, show_shapes=True, to_file=os.path.join(p_save, 'model_graph.png'))

        # fit stacked model
        y_train_enc = to_categorical(y_train_b)
        y_val_enc = to_categorical(y_val)
        hist = stacked_model.fit(x_train_b, y_train_enc,
                                 validation_data=(x_val, y_val_enc),
                                 batch_size=batchSize, epochs=nEpoch)

        # save stacked model and training history
        stacked_model.save(os.path.join(p_save, 'stacked_nn_seed_' + str(seed)+'_lnpo' + str(lnpoi) + '.h5'))
        pd.DataFrame(hist.history).to_csv(os.path.join(p_save, 'history_seed_'+ str(seed)+'lnpo' + str(lnpoi) + '.csv'))

        # make predictions and evaluate
        res = predict_stacked_model(stacked_model, x_val, y_val)
        report = add_to_report('stacked_nn', seed, lnpoi, 'val', res, report)
        res = predict_stacked_model(stacked_model, x_train_b, y_train_b)
        report = add_to_report('stacked_nn', seed, lnpoi, 'train', res, report)


        ### stacked by Logistric Regression  ###

        lr_model = LogisticRegression().fit(x_train_b, y_train_b)
        pickle.dump(lr_model, open(os.path.join(p_save, 'stacked_lr_seed_' + str(seed)+'_lnpo' + str(lnpoi) +'.sav'), 'wb'))

        res = predict_logistic_model(lr_model, x_val, y_val)
        report = add_to_report('stacked_lr', seed, lnpoi, 'val', res, report)
        res = predict_logistic_model(lr_model, x_train_b, y_train_b)
        report = add_to_report('stacked_lr', seed, lnpoi, 'train', res, report)

        report.to_csv(os.path.join(p_save, 'report.csv'))

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)


# to visulize lr models
if False:
    p_save = os.path.join('history', 'stacking_ensemble', '20201022_022901')
    for seed in seeds:
        for lnpoi in lnpo:
            loaded_model = pickle.load(open(os.path.join(p_save, 'stacked_lr_seed_' + str(seed)+'_lnpo' + str(lnpoi) +'.sav'), 'rb'))

            if seed == seeds[0] and lnpoi == lnpo[0]:
                tp = loaded_model.coef_
                tp2 = list(loaded_model.intercept_)
            else:
                tp = np.vstack((tp, loaded_model.coef_))
                tp2.extend(list(loaded_model.intercept_))

    print(np.mean(tp, axis = 0))
    print(np.mean(tp2))
    print(features)

