# get data labels for each EEG.urevent of study 2

import os
import numpy as np
import pandas as pd
from sys import exit
import scipy.io as sio


f_main = os.path.join(os.path.dirname(os.getcwd()), '3data2')


def get_beh(sub, task, nBack = 0,  printOn = True):  # nBack == 0 - load all

    if task.lower() == 'sart':
        data = get_data_sart(sub)
    elif task.lower() == 'vs':
        data = get_data_vs(sub)
    else:
        exit('ERROR: Invalid task')

    if nBack == 0:
        if printOn: print('Load all behavioral data.')
    elif nBack > 0:
        if printOn: print('Load data from previous %s trials.' %nBack)
        data = get_data_nbk(data, nBack)
    else:
        exit('Invalid nBack')

    data = add_labs(data)

    return data


def add_labs(data):
    df = data.copy()
    df['state'] = np.nan
    df['state'] = np.where(df['pr_resp']>0, 'ot', 'mw')
    df.loc[df['pr_resp']==0,'state'] = 'uc'
    return df


def get_data_nbk(data, n):
    df = data[data['dis2pr'] < n]
    return df


def get_data_vs(sub, ureventOn = True):

    filename = os.path.join(f_main, 'raw', str(sub) + '_vs.csv')
    df = pd.read_csv(filename)

    cols2analyze = ['block', 'trigger', 'target', 'blocks.thisRepN', 'blocks.thisN', 'trials.thisN',
                    'nTri', 'nSqr', 'nPen', 'nHex', 'resp.keys', 'resp.corr', 'resp.rt',
                    'rating.response', 'rating.rt', 'group']
    df = df.filter(cols2analyze)
    # filter prac trials
    # this last line where blocks.thisN==NaN should remain
    startRowId = df.index[df['blocks.thisN'] == 0].tolist()[0]
    df = df[startRowId:]
    df = add_dis2pr(df)
    if ureventOn:
        df = add_urevent(df, sub)

    return df


def get_data_sart(sub, ureventOn = True):

    filename = os.path.join(f_main, 'raw', str(sub)+'_sart.csv')
    df = pd.read_csv(filename)
    cols2analyze = ['number', 'type', 'trigger', 'blocks.thisN', 'trials.thisN',
                    'resp.keys', 'resp.corr', 'resp.rt', 'rating.response', 'rating.rt']
    df = df.filter(cols2analyze)
    rmRows = df['rating.response'].isnull() & df['resp.corr'].isnull()
    df = df[~rmRows]
    df = add_dis2pr(df)
    if ureventOn:
        df = add_urevent(df, sub)

    return df


def add_dis2pr(df):

    data = df.copy()
    data['dis2pr'] = np.nan
    data['pr_resp'] = np.nan
    data['pr_rt'] = np.nan

    blocks = data['blocks.thisN'].unique()
    blocks = blocks[~pd.isnull(blocks)]

    probes = data.filter(['rating.response', 'rating.rt'])
    probes = probes.dropna()

    # check
    if len(blocks) != probes.shape[0]:
        exit('ERROR: Unequal number of blocks and probes!')

    for bi in range(len(blocks)):
        block = blocks[bi]
        rows = data['blocks.thisN'] == block
        trials = data[rows]['trials.thisN']
        data.loc[rows,'dis2pr'] = trials.shape[0] - trials - 1
        data.loc[rows, 'pr_resp'] = probes.iloc[bi][0]
        data.loc[rows, 'pr_rt'] = probes.iloc[bi][1]

    return data


def add_urevent(df, sub):
    data = df.copy()
    data['urevent'] = np.nan
    triggers = data['trigger'].dropna().unique()

    f_load = os.path.join(f_main, 'urevent_matfile', str(sub) + '.mat')
    eeg_triggers = sio.loadmat(f_load)['urevent'][0]
    urevents =  np.in1d(eeg_triggers, triggers) # filter out triggers for the current study

    # consitency check:
    # in case the eeg recording was started after the behavior recording
    len2cut = len(data['trigger'].dropna()) - len(eeg_triggers[urevents])
    if len2cut >= 0:
        if not np.array_equal(data[len2cut:]['trigger'].dropna(), eeg_triggers[urevents]):
            exit('ERROR: Unmatched triggers in behavioral and EEG files')
    else:
        exit('ERROR: Unmatched triggers in behavioral and EEG files')

    rows = np.in1d(data['trigger'], triggers)
    rows[:len2cut] = False
    data.loc[rows, 'urevent'] = np.nonzero(urevents)[0]+1  # the first position == urev 1

    return data






