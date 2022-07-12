# cancel the downsampling of the raw
import scipy.io as sio
import os
import numpy as np
from code_pyfile.getLabs1 import get_beh
from code_pyfile.normalize import normalize
from sklearn.model_selection import train_test_split
from math import floor
from tqdm import tqdm
from collections import Counter
import cupy as cp



#nBack = 3  # based on reports: pos - 'ot', neg - 'mw', 0 - 'uc'
#conds = ['ot', 'mw']  # ordered as number format labels

locs16On = False
plocs16On = True
flocsOn = False  # downsamplizng frequencies
# note after subsetting chans/chanpairs, the same scaling method would result differently


if locs16On:
    locs = sio.loadmat('pars_stFeats.mat')['locs16_32'][0] - 1

if plocs16On:
    plocs = sio.loadmat('pars_stFeats.mat')['plocs16_32'].T[0] - 1

if flocsOn:
    flocs = sio.loadmat('pars_stFeats.mat')['flocs'].T[0] - 1


def get_subs():
    subs = range(1, 31)
    return subs


# load data: x.shape = (nTrial, nChan, nPnt)
# if norm is On, normalized within each session
def load_dataset(sub, task, feat, conds, nBacks, norm = False, sidx = 'all',
                 gpu2use = -1, sessions = [1,2]):

    for ss in sessions:
        
        ci = 0  # ci can be smaller than condi if some classes are missing in one dataset
        for condi in range(len(conds)):
            
            if isinstance(nBacks, list):
                beh = get_beh(sub, task, nBacks[condi], [ss])
            else:
                beh = get_beh(sub, task, nBacks, [ss])

            cond = conds[condi]
            triallist = beh[beh['state']==cond][['urevent','session']]
            ntrial = triallist.shape[0]

            if ntrial == 0:
                continue

            ti = 0  # ti can be smaller than triali if some trials have been skipped
            for triali in range(ntrial):

                trialname = triallist.iloc[triali,0]
                trialname = str(int(trialname)).zfill(4)
                if triallist.iloc[triali,1] == 1:
                    trialname = trialname + '0000'
                else:
                    trialname = '0000' + trialname
                f_load = os.path.join('.', 'feats_matfile','1', str(sub).zfill(3), trialname, feat+ '.mat')

                if os.path.exists(f_load):  # skip noisy trials, which are removed during preprocessing
                    mat = sio.loadmat(f_load)['data']
                else:
                    continue

                # downsample some features
                if feat == 'wst':
                    mat = mat[::10,::4,:]

                # subset by frequencies
                if (feat == 'power' or feat == 'ispc') and flocsOn:
                    mat = mat.copy()[flocs,:, :]

                # subset by channels (locs16On, plocs16On)
                if feat == 'raw' and locs16On:
                    mat = mat.copy()[locs,:]
                elif feat == 'power' and locs16On:
                    mat = mat.copy()[:, :, locs]
                elif feat == 'ispc' and plocs16On:
                    mat = mat.copy()[:, :, plocs]
                elif feat == 'wst' and locs16On:
                    mat = mat.copy()[:, :, locs]

                # subset by the specified spatial indices (sidx, list)
                if isinstance(sidx, list):
                    if feat == 'raw': mat = mat.copy()[sidx, :]
                    else: mat = mat.copy()[:,:,sidx]

                if ti == 0:
                    if feat == 'raw':
                        x_temp = np.zeros((1, mat.shape[0], mat.shape[1]), dtype = '<f4')
                    else:
                        x_temp = np.zeros((1, mat.shape[0], mat.shape[1], mat.shape[2]), dtype='<f4')

                    x_temp[0,] = mat.copy()
                    y_temp = np.full((1,), condi, dtype='uint8')

                    if gpu2use > -1:
                        x_temp = cp.asarray(x_temp)
                        y_temp = cp.asarray(y_temp)

                else: # ti > 0
                    if gpu2use > -1:
                        x_temp = cp.append(x_temp, [mat.copy()], axis = 0)
                        y_temp = cp.append(y_temp, condi)
                    else:
                        x_temp = np.append(x_temp, [mat.copy()], axis = 0)
                        y_temp = np.append(y_temp, condi)

                ti += 1

            if ti == 0: # if trials in the triallist are all removed during preprocessing
                continue

            if ci == 0:
                x = x_temp.copy()
                y = y_temp.copy()
            else:
                if gpu2use == -1:
                    x = np.concatenate((x, x_temp.copy()))
                    y = np.concatenate((y, y_temp.copy()))
                else:
                    x = cp.concatenate((x, x_temp.copy()))
                    y = cp.concatenate((y, y_temp.copy()))

            ci += 1

        # normalize intra-session
        if isinstance(norm, str):
            x = normalize(x, norm)

        if ss == sessions[0]:
            x_all = x.copy()
            y_all = y.copy()
        else:
            if gpu2use == -1:
                x_all = np.concatenate((x_all, x.copy()))
                y_all = np.concatenate((y_all, y.copy()))
            else:
                x_all = cp.concatenate((x_all, x.copy()))
                y_all = cp.concatenate((y_all, y.copy()))

    return x_all, y_all



def load_dataset_feats(sub, task, feats, conds, nBacks, norm = False, sidx = 'all', gpu2use = -1):
# only works when combining power and ispc, since they share the same freq/pnt (1,2) dimensions

    for fi in range(len(feats)):
        feat = feats[fi]
        f_sidx = sidx[fi]
        x, y = load_dataset(sub, task, feat, conds, nBacks, norm, f_sidx, gpu2use)

        if fi == 0:
            xs = x.copy()
            ys = y.copy()
        else:
            xs = np.concatenate((xs, x.copy()), axis = 3)  # concatenate at the last (chan/chanpair) dimension

    return xs, ys



def load_dataset_n(subs, task, feats, conds, nBacks, norm = False, sidx = 'all', gpu2use = -1):

    s = []
    for sub in tqdm(subs):
        x, y = load_dataset_feats(sub, task, feats, conds, nBacks, norm, sidx, gpu2use)

        if sub == subs[0]:
            xs = x.copy()
            ys = y.copy()
        else:
            xs = np.concatenate((xs, x.copy()))
            ys = np.concatenate((ys, y.copy()))
        s.extend([sub]*x.shape[0])
        print('Load data of SUB', sub, ' Trial count: ', x.shape[0], 'Total trial count: ', xs.shape[0])

    return xs, ys, s



def load_dataset_n_tasks(subs, tasks, feats, conds, winlens, norm = False, sidx='all', gpu2use = -1):

    for ti in range(len(tasks)):

        task = tasks[ti]
        nBacks = []
        for ci in range(len(conds)):
            if isinstance(winlens, list):
                nBacks.append(get_nBack(task, winlens[ti][ci]))
            else:
                nBacks.append(get_nBack(task, winlens))
        
        x, y, s = load_dataset_n(subs, task, feats, conds, nBacks, norm, sidx, gpu2use)
        if task == tasks[0]:
            xs = x.copy()
            ys = y.copy()
            ss = s.copy()
        else:
            xs = np.concatenate((xs, x.copy()))
            ys = np.concatenate((ys, y.copy()))
            ss.extend(s)
        print('Load data of task', task, ' Trial count: ', x.shape[0], 'Total trial count: ', xs.shape[0])

    return xs, ys, ss



def load_dataset_n_split(subs, task, feats, conds, nBacks, norm = False, testSize = 0.2, randSeed = None, sidx = 'all', gpu2use=-1):

    for sub in tqdm(subs):
        x, y = load_dataset_feats(sub, task, feats, conds, nBacks, norm, sidx, gpu2use)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize, random_state = randSeed)

        if sub == subs[0]:
            xs_train = x_train.copy()
            xs_test = x_test.copy()
            ys_train = y_train.copy()
            ys_test = y_test.copy()
        else:
            xs_train = np.concatenate((xs_train, x_train.copy()))
            xs_test = np.concatenate((xs_test, x_test.copy()))
            ys_train = np.concatenate((ys_train, y_train.copy()))
            ys_test = np.concatenate((ys_test, y_test.copy()))

        print('Load data of SUB', sub,
              ' Training Trial Count:', x_train.shape[0], '',
              ' Test Trial Count:', x_test.shape[0], '',
              ' Total Training Trial Count:', xs_train.shape[0], '',
              ' Total Test Trial Count:', xs_test.shape[0], ''
              )

    return xs_train, xs_test, ys_train, ys_test


def get_nBack(task, winlen): # winlen in seconds

    if task == 'vs':
        nBack = floor(winlen/4.8)
    elif task == 'sart':
        nBack = floor(winlen/6)

    return nBack



def split_subs(subs, testSize = 0.2, randSeed = None):
    np.random.seed(randSeed)
    subs = list(subs)
    np.random.shuffle(subs)
    nSubs = len(subs)
    nTest = round(nSubs * testSize)
    subs_test = subs[:nTest]
    subs_train = subs[nTest:]

    return subs_train, subs_test


def get_class_size(subs, tasks, conds, winlens, gpu2use = -1):
    x,y,s = load_dataset_n_tasks(subs, tasks, ['raw'], conds, winlens, norm=False, sidx='all', gpu2use=gpu2use)# load the smallest feature for counting trials
    return (Counter(y))




# x = normalize(x, unit = 'signal')
# for check only
#x_bk = x.copy()
#x = x_bk[:,:28,:28]

# for check only
#x0 = x[y==0,:,:]
#x1 = x[y==1,:,:]

#import matplotlib.pyplot as plt
#plt.subplot(1,2,1)
#plt.imshow(x0.mean(axis = 0), vmin =0 , vmax = 1, aspect = 'auto')
#plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(x1.mean(axis = 0), vmin =0 , vmax = 1, aspect = 'auto')
#plt.colorbar()

