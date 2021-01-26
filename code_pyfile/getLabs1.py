# get data labels for each EEG.urevent of study 2

import os
import numpy as np
import pandas as pd
from sys import exit
import scipy.io as sio


f_main = os.path.join(os.path.dirname(os.getcwd()), '3data')


def get_beh(sub, task, nBack = 0, session = [1,2], printOn = True):  # nBack == 0 - load all

    if np.isin(task.lower(), ['sart', 'vs']):
        data = get_data(sub, task, True, session)
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
    prs = pd.to_numeric(df['pr_content'])
    df.loc[df['pr_content'] == 1, 'state'] = 'ot'
    df.loc[df['pr_content'] == 2, 'state'] = 'ot'
    df.loc[df['pr_content'] == 3, 'state'] = 'mw'
    df.loc[df['pr_content'] == 4,'state'] = 'ds'
    df.loc[df['pr_content'] == 5, 'state'] = 'mw'
    df.loc[df['pr_content'] == 6,'state'] = 'uc'
    return df


def get_data_nbk(data, n):
    df = data[data['dis2pr'] < n]
    return df


def get_data(sub, task, ureventOn = True, session = [1,2]):

    for ssi in range(len(session)):

        ss = session[ssi]
        filename = os.path.join(f_main, 'raw', 'beh', 'subject-'+str(sub) + '_s'+str(ss)+'.csv')
        df = pd.read_csv(filename)

        cols2analyze = ["code", "correct_keyboard_resp", "correct_keyboard_response1",
                        "correct_keyboard_response2", "correct_keyboard_response3", "probe_on",
                        "response_probe_content", "response_probe_orientation",
                        "response_probe_stickness", "response_probe_valence",
                        "response_time_keyboard_resp", "response_time_keyboard_response1",
                        "response_time_keyboard_response2", "response_time_keyboard_response3",
                        "stimulus", "task"]
        df = df.filter(cols2analyze)

        df = add_dis2pr(df)

        if ureventOn:
            df = add_urevent(df, sub, ss)

        df['session'] = ss

        if ssi == 0:
            data = df.copy()
        else:
            data = pd.concat([data, df], sort=False)

        # subset for clean trials
        data = data[data['remain']==1]

        # subset for task
        data = data[data['task'] == task]

    return data


def add_dis2pr(df):

# only labelled by content
    data = df.copy()
    data['dis2pr'] = np.nan
    data['pr_content'] = np.nan

    probe_pos = data['probe_on'] == 1
    probe_pos = np.array(range(data.shape[0]))[probe_pos]

    for pi in range(len(probe_pos)):

        pr_now = probe_pos[pi]
        if pi == 0:
            rows = range(pr_now+1)
        else:
            pr_prev = probe_pos[pi-1]
            rows = range(pr_prev+1, pr_now+1)

        data.loc[rows, 'dis2pr'] = pr_now - rows
        pr_resp_c = data.loc[pr_now, 'response_probe_content']
        if isinstance(pr_resp_c, int):
            data.loc[rows, 'pr_content'] = pr_resp_c
        else:
            try:
                data.loc[rows, 'pr_content'] = int(pr_resp_c)
            except ValueError:
                continue
                #print('Invalid responses to probe. Skip')

    return data


def add_urevent(df, sub, session):
    data = df.copy()
    data['urevent'] = np.nan
    data['remain'] = np.nan
    triggers = [20, 21, 10, 11, 12, 13, 14]

    f_load = os.path.join(f_main, 'urevent_matfile', str(sub) + '_'+ str(session)+'.mat')
    mat = sio.loadmat(f_load)['mat']
    eeg_triggers = mat[:,0]
    urevents =  np.in1d(eeg_triggers, triggers) # filter out triggers for the current study

    # consitency check:

    if not sum(urevents) == data.shape[0]:
        exit('ERROR: Unmatched triggers in behavioral and EEG files')
    else:

        # only check for those of vs as triggers sent in the sart were not recorded in the beh.csv files
        # (which were created as temporary variables)
        rows2match = data['task'] == 'vs'
        if not np.array_equal(data[rows2match].code, eeg_triggers[urevents][rows2match]):
            exit('ERROR: Unmatched triggers in behavioral and EEG files')

    data['urevent'] = np.nonzero(urevents)[0]+1  # the first position == urev 1
    data['remain'] = mat[urevents,1]

    return data






