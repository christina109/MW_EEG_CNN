
# load trained models for across-exp predictions

testOn = False # f_hist will be a temporary folder
# device number
gpu2use = -1

if gpu2use > -1:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu2use], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
else:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras
from sklearn.metrics import classification
import pickle_report
import numpy as np
import pandas as pd
import os
import code_pyfile.loadData1 as ld1
import code_pyfile.loadData2 as ld2
from tqdm import tqdm
import time

t0 = time.perf_counter()
###   settings   ###

# randomization
nn = 'simplecnn_5'
seeds = [36,54,87]
lnpo = range(5)
stacking_models = ['nn', 'lr']
f_model = '20201022_022901' # folder name
p_stack = os.path.join('history', 'stacking_ensemble', f_model)  # model name
features = pd.read_csv(os.path.join(p_stack, 'features.csv'))

# testing dataset parameters
studies = [2]
winlen = 20  # in seconds
nClass = 2
conds = ['ot', 'mw']
tasks = ['vs', 'sart']  # 'vs','sart'

# save setting
p_save =  os.path.join('.', 'history', 'asp', f_model)
if not os.path.exists(p_save):
    os.makedirs(p_save)


# config normalizations
def get_norm(feat):
    if feat == 'raw':
        norm = 'off'
    elif feat == 'power' or feat == 'ispc' or feat == 'wst':
        norm = 'off'
    return norm


def get_subs(study, testOn=False):
    if not testOn:
        if study == 1:
            subs = range(1,31)
        else:
            subs = range(301,331)
    else:  # make sure enough particitpants for test
        if study == 1:
            subs = range(1,6)
        else:
            subs = range(301,306)
    return subs



# load models and data
def load_models_and_data(study, seed, lnpoi):

    all_x = list()
    all_y = list()

    for ri in features.index:
        feat = features.iloc[ri,0]
        hpar = features.iloc[ri,1]
        hvals = features.iloc[ri,2]
        hvals = eval(hvals)  # remove quotes

        norm = get_norm(feat)
        subs = get_subs(study)

        for hval in hvals:

            p_feat = os.path.join('history', nn + '_' + feat, hpar + '_' + str(seed))

            # load data
            if study == 1:
                if hpar == 'channel':
                    x, y, _ = ld1.load_dataset_n_tasks(subs=subs, tasks=tasks, feats=[feat], conds=conds,
                                                       winlen=winlen, norm=norm, sidx=[[hval]])
                else:
                    x, y, _ = ld1.load_dataset_n_tasks(subs=subs, tasks=tasks, feats=[feat], conds=conds,
                                                       winlen=winlen, norm=norm)
            else:
                if hpar == 'channel':
                    x, y, _ = ld2.load_dataset_n_tasks(subs=subs, tasks=tasks, feats=[feat], conds=conds,
                                                       winlen=winlen, norm=norm, sidx=[[hval]])
                else:
                    x, y, _ = ld2.load_dataset_n_tasks(subs=subs, tasks=tasks, feats=[feat], conds=conds,
                                                       winlen=winlen, norm=norm)

            if feat == 'raw':
                x = np.transpose(x, (0, 2, 1))

            # load model
            p_model = os.path.join('history', nn + '_' + feat, hpar + '_' + str(seed), str(hval))
            with open(os.path.join(p_model, 'model_init.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(p_model, 'weights_lnpo' + str(lnpoi) + '.h5'))
            print('>loaded %s' % p_model)

            # get outcome
            yhat = model.predict(x)

            # outcome from each model concatenated as the new input
            if all_x == []:
                all_x = yhat.copy()
            else:
                all_x = np.concatenate((all_x, yhat), axis = 1)

            # only one 'y' is required
            if all_y == []:
                all_y = y.copy()
            else:
                if not np.array_equal(all_y, y):
                    import sys
                    sys.exit('Inputs not from the same trials')

    return all_x, all_y


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
for study in tqdm(studies):
    for seed in tqdm(seeds):
        for lnpoi in tqdm(lnpo):

            x_test, y_test = load_models_and_data(study, seed, lnpoi)

            # TEST ONLY: fake other dimensions for matching the input data size - reduce loading time
            if False:
                tp = np.tile(x_test, (1,4))
                x_test = tp

            for stacking_model in stacking_models:

                if stacking_model == 'nn':
                    f_load = (os.path.join(p_stack, 'stacked_nn_seed_' + str(seed) + '_lnpo' + str(lnpoi) + '.h5'))
                    model = keras.models.load_model(f_load)
                    res = predict_stacked_model(model, x_test, y_test)
                else:
                    f_load = (os.path.join(p_stack, 'stacked_lr_seed_' + str(seed) + '_lnpo' + str(lnpoi) + '.sav'))
                    model = pickle.load(open(f_load, 'rb'))
                    res = predict_logistic_model(model, x_test, y_test)

                report =  add_to_report(stacking_model, seed, lnpoi, study, res, report)
                report.to_csv(os.path.join(p_save, 'report.csv'))

t2 = time.perf_counter()
print("Time elapsed [hr]: ", (t2-t0)/3600) # CPU seconds elapsed (floating point)