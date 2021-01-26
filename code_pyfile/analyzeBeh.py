import os
import numpy as np
import pandas as pd
import code_pyfile.getLabs1 as gl1
import code_pyfile.getLabs2 as gl2
from math import floor
from itertools import compress
import sys
from collections import Counter



def count_trials(studies, subs, tasks, winlen):  # report ratio as well
    study = []
    sub = []
    task = []
    n_ot = []
    n_mw = []

    toolbar_width = len(studies)*len(subs)*len(tasks)

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    for sd in studies:
        for s in subs:
            for t in tasks:

                nBack = get_nBack(sd, t, winlen)
                if sd == 1:
                    data = gl1.get_beh(s, t, nBack, printOn=False)
                else:
                    data = gl2.get_beh(s, t, nBack, printOn=False)

                study.append(sd)
                sub.append(s)
                task.append(t)
                n_ot.append(sum(data['state'] == 'ot'))
                n_mw.append(sum(data['state'] == 'mw'))

                # update the bar
                sys.stdout.write("-")
                sys.stdout.flush()
    sys.stdout.write("]\n")  # this ends the progress bar

    df = pd.DataFrame({'study': study,
                       'task': task,
                       'sub': sub,
                       'n_ot': n_ot,
                       'n_mw': n_mw})
    df.to_csv('trial_count.csv')
    print('saved.')




def report_beh(studies, subs, tasks, winlen):
    study = []
    sub = []
    task = []
    acc = []
    rt = []

    toolbar_width = len(studies)*len(subs)*len(tasks)

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    for sd in studies:
        for s in subs:
            for t in tasks:
                tp = analyze_beh(sd, s, t, winlen)
                study.append(sd)
                sub.append(s)
                task.append(t)
                acc.append(tp[0])
                rt.append(tp[1])

                # update the bar
                sys.stdout.write("-")
                sys.stdout.flush()
    sys.stdout.write("]\n")  # this ends the progress bar

    df = pd.DataFrame({'study': study,
                       'task': task,
                       'sub': sub,
                       'acc': acc,
                       'rt': rt})
    df.to_csv('beh_report.csv')
    print('saved.')




def analyze_beh(study, sub, task, winlen):

    nBack = get_nBack(study, task, winlen)
    if study == 1:
        data = gl1.get_beh(sub, task, nBack, printOn = False)
    else:
        data = gl2.get_beh(sub, task, nBack, printOn=False)

    if task == 'vs':
        accuracy = np.average(data.correct_keyboard_resp)
        rt = np.average(data.response_time_keyboard_resp)
    else:  # task == 'sart'
        correct = []
        response_time = []

        tp = data[data.stimulus=='T']
        tp_arr = np.logical_and(tp.correct_keyboard_response1, tp.correct_keyboard_response2)
        tp_arr = np.logical_and(tp_arr, tp.correct_keyboard_response3)
        correct.extend(tp_arr)

        tp = data[data.stimulus!='T']
        tp_arr = np.logical_or(tp.correct_keyboard_response1, tp.correct_keyboard_response2)
        tp_arr = np.logical_or(tp_arr, tp.correct_keyboard_response3)
        correct.extend(tp_arr)

        for ri in range(tp.shape[0]):
            row = tp.iloc[ri,]
            if row.response_time_keyboard_response1 < 295:
                response_time.append(row.response_time_keyboard_response1)
            elif row.response_time_keyboard_response2 < 895:
                response_time.append(row.response_time_keyboard_response2+295)
            elif row.response_time_keyboard_response3 < 2995:
                response_time.append(row.response_time_keyboard_response3 + 295 + 895)
            else:
                response_time.append(0)  # no response

        #subset for correct trials
        tp_rt = list(compress(response_time, correct))

        accuracy = np.average(correct)
        rt = np.average(tp_rt)

    return [accuracy, rt]





def ifelse(statement, a, b):
    if statement:
        return a
    else:
        return b




def get_nBack(study, task, winlen):  # winlen in seconds , study 1
    if study == 1:

        if task == 'vs':
            nBack = floor(winlen / 4.8)
        elif task == 'sart':
            nBack = floor(winlen / 6)
    else:
        if task == 'vs':
            nBack = floor(winlen / 8.654)
        elif task == 'sart':
            nBack = floor(winlen / 3)

    return nBack
