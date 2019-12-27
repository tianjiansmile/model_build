#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Dan'

import datetime
import numpy as np
import pandas as pd
from numpy import nanmin, nanmax, nanmean, nanmedian, nanstd, nansum
from scipy.stats import wasserstein_distance

missingValue = -1
errorValue = -2


def wrapper_div(a, b):
    a = wrapper_float(a)
    b = wrapper_float(b)
    try:
        if b is None or b == 0:
            return missingValue
        else:
            return a/b
    except ValueError or TypeError:
        return errorValue


def wrapper_min(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nanmin(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_max(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nanmax(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_mean(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nanmean(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_median(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nanmedian(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_std(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nanstd(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_sum(ls):
    ls = [wrapper_float(l) for l in ls]
    if len(ls) == 0:
        return 0
    try:
        return nansum(ls)
    except ValueError or TypeError:
        return errorValue


def wrapper_emd(hist0, hist1):
    """
    NOTE: for memory friendly, do not use pyemd!
    You can also calculate the EMD directly from two arrays of observations:emd_samples.
    :param hist0: np.array or list
    :param hist1: np.array or list
    :return:
    """
    try:
        hist0 = [wrapper_float(k) for k in hist0]
        hist1 = [wrapper_float(k) for k in hist1]
        hist0 = np.array(hist0) if isinstance(hist0, list) else hist0
        hist1 = np.array(hist1) if isinstance(hist1, list) else hist1
        # if which_emd == 'pyemd':
        #     rlt = pyemd.emd_samples(np.array(hist0), np.array(hist1))
        return wasserstein_distance(np.array(hist0), np.array(hist1))
    except ValueError or TypeError:
        return errorValue


def apply_basic_metrics(ls, feat_prefix, cnt=False):
    rlt = dict()
    rlt['{feat_prefix}_min'.format(feat_prefix=feat_prefix)] = wrapper_min(ls)
    rlt['{feat_prefix}_max'.format(feat_prefix=feat_prefix)] = wrapper_max(ls)
    rlt['{feat_prefix}_mean'.format(feat_prefix=feat_prefix)] = wrapper_mean(ls)
    rlt['{feat_prefix}_median'.format(feat_prefix=feat_prefix)] = wrapper_median(ls)
    rlt['{feat_prefix}_std'.format(feat_prefix=feat_prefix)] = wrapper_std(ls)
    rlt['{feat_prefix}_sum'.format(feat_prefix=feat_prefix)] = wrapper_sum(ls)
    if cnt is True:
        rlt['{feat_prefix}_count'.format(feat_prefix=feat_prefix)] = len(ls)
    return rlt


def apply_ratio_basic_metrics(feat, nume, deno, cnt=False):
    ratio_feat = dict()
    ratio_feat_prefix = '{nume}_vs_all_Ratio'.format(nume=nume)
    ls_metrics = ['min', 'max', 'mean', 'median', 'std', 'sum']
    if cnt is True:
        ls_metrics.append('cnt')
    for metric in ls_metrics:
        ratio_feat['{rfp}_{metric}'.format(rfp=ratio_feat_prefix, metric=metric)] = \
            wrapper_div(feat.get('{nume}_{metric}'.format(nume=nume, metric=metric)),
                        feat.get('{deno}_{metric}'.format(deno=deno, metric=metric)))
    return ratio_feat


def wrapper_missing_value(my_dict):
    # convert missing values to -1
    for k, v in my_dict.items():
        if v in [None, 'null', '']:
            my_dict[k] = missingValue
    return my_dict


def wrapper_strptime(s, stype='YmdHMS'):
    assert stype in ['YmdHMS', 'Ymd', 'Ym']
    if s is None:
        return None
    try:
        if stype == 'YmdHMS':
            rlt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        elif stype == 'Ymd':
            rlt = datetime.datetime.strptime(s, '%Y-%m-%d')
        else:
            rlt = datetime.datetime.strptime(s, '%Y-%m')
    except ValueError or TypeError:
        rlt = None
    return rlt


def wrapper_diff_time(df, col_start, col_end, metric):
    """
    :param df: data frame
    :param col_start: type of this column is datetime
    :param col_end: type of this column is datetime
    :param metric: hour or day
    :return:
    """
    assert metric in ['day', 'hour']
    rlt = []
    for k in range(len(df)):
        A = df[col_start].iloc[k]
        B = df[col_end].iloc[k]
        try:
            dif = B - A
            if metric == 'day':
                rlt.append(dif.days)
            else:
                rlt.append(dif.days * 24 + dif.seconds//3600)
        except TypeError:
            rlt.append(0)  # Note: no time replace by 0 since
    return rlt


def convert_ls_to_df(ls, cols):
    import time
    starttime = time.time()
    df = pd.DataFrame(columns=cols)
    for l in ls:
        val_line = []
        for col in cols:
            val_line.append(l.get(col))
        # df.loc[len(df)] = val_line

    endtime = time.time()
    print(' cost time: ', endtime - starttime)
    return df

def convert_ls_to_df_better(ls, cols):

    df = pd.DataFrame(columns=cols)
    all_val_line = []
    for l in ls:
        val_line = []
        for col in cols:
            val_line.append(l.get(col))
        all_val_line.append(val_line)

    # 将嵌套列表，按照对应列名转换成为Dataframe
    df = pd.DataFrame(all_val_line,columns=cols)

    return df


def wrapper_float(v):
    try:
        rlt = float(v)
    except:
        rlt = missingValue
    return rlt


def wrapper_len(ls):
    ls = [l for l in ls if l is not None]
    return len(np.unique(ls))
