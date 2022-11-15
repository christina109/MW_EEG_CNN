import scipy.io as sio
import os
import numpy as np
from code_pyfile.scaling import normalize
from tqdm import tqdm
from collections import Counter

srate_raw = 256
srate_power = 50

def replace_labels(x):
    if x=='ot':
        return 0
    if x=='mw':
        return 1

# for test
sub = 1
sessions = range(1,12)
feat = 'power'
conds = ['ot', 'mw']
winlen = 2
norm = 'chan'
sidx = list(range(20))
gpu2use = -1

subs = [1,2]
feats = ['raw']

# load data: x.shape = (nTrial, nChan, nPnt)
# if norm is On, normalized within each session
def load_dataset(sub, sessions, feat, conds, winlen, norm = False, sidx = 'all',
                 gpu2use = -1):
    if gpu2use > -1:
        import cupy as cp

    conds_bk = conds.copy()
    conds = list(map(replace_labels, conds_bk))
    for ss in tqdm(sessions):
        p_load = os.path.join('.', 'feats_matfile', 'Grandchamp', str(sub), str(ss).zfill(2))
        ntrial = len(os.listdir(p_load))

        for triali in range(ntrial):
            trialname = str(triali+1).zfill(4)
            cond = sio.loadmat(os.path.join(p_load, trialname, 'label.mat'))['data'][0][0]
            if cond not in conds:
                continue

            f_load = os.path.join(p_load, trialname, feat+ '.mat')
            mat = sio.loadmat(f_load)['data']

            if feat == 'raw':  # segment the initial 8s epoch
                mat = mat.reshape(mat.shape[0], winlen*srate_raw, int(8/winlen))
            else:
                mat = np.transpose(mat, (0,2,1))
                mat = mat.reshape(mat.shape[0], mat.shape[1], winlen * srate_power, int(8 / winlen))

            # downsample some features
            if feat == 'raw':
                mat = mat[:,::2]

            # subset by the specified spatial indices (sidx, list)
            if isinstance(sidx, list):
                if feat == 'raw':
                    mat = mat.copy()[sidx, :,:]
                else:
                    mat = mat.copy()[:,sidx,:,:]

            if feat == 'raw':
                mat = np.transpose(mat, (2,0,1))
            else:
                mat = np.transpose(mat, (3,0,2,1))

            if triali == 0:
                y = [cond] * mat.shape[0]
                x = mat.copy()
                if gpu2use > -1:
                    x = cp.asarray(x)
            else: # triali > 0
                y.extend([cond] * mat.shape[0])
                if gpu2use > -1:
                    x = cp.concatenate((x, mat.copy()))
                else:
                    x = np.concatenate((x, mat.copy()))

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



def load_dataset_feats(sub, sessions, feats, conds, winlen, norm = False, sidx = 'all', gpu2use = -1):
# only works when combining power and ispc, since they share the same freq/pnt (1,2) dimensions

    for fi in range(len(feats)):
        feat = feats[fi]
        f_sidx = sidx[fi]
        x, y = load_dataset(sub, sessions, feat, conds, winlen, norm, f_sidx, gpu2use)

        if fi == 0:
            xs = x.copy()
            ys = y.copy()
        else:
            xs = np.concatenate((xs, x.copy()), axis = 3)  # concatenate at the last (chan/chanpair) dimension

    return xs, ys



def load_dataset_n(subs, sessions, feats, conds, winlen, norm = False, sidx = 'all', gpu2use = -1):

    s = []
    for sub in tqdm(subs):
        x, y = load_dataset_feats(sub, sessions, feats, conds, winlen, norm, sidx, gpu2use)

        if sub == subs[0]:
            xs = x.copy()
            ys = y.copy()
        else:
            xs = np.concatenate((xs, x.copy()))
            ys = np.concatenate((ys, y.copy()))
        s.extend([sub]*x.shape[0])
        print('Load data of SUB', sub, ' Trial count: ', x.shape[0], 'Total trial count: ', xs.shape[0])

    return xs, ys, s


def split_subs(subs, testSize = 0.2, randSeed = None):
    np.random.seed(randSeed)
    subs = list(subs)
    np.random.shuffle(subs)
    nSubs = len(subs)
    nTest = round(nSubs * testSize)
    subs_test = subs[:nTest]
    subs_train = subs[nTest:]

    return subs_train, subs_test


