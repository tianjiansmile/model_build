# scorecard analysis
# 20190815
# lh create

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats.stats as stats
from numpy import *
from pandas import *
import seaborn as sns

class analysis():
    def __init__(self,):
        pass

    def MissingValue(self,var, data):
        ### 处理缺失值
        var_missing = var.iloc[where(var.isnull() == True)]
        missing_percent = var_missing.shape[0] / var.shape[0]
        missing_total_num = var_missing.shape[0]
        missing_info = []
        missing_info.append('missing value')
        missing_info.append(missing_total_num)

        missing_info = DataFrame([missing_info], columns=data.columns)
        data = data.append(missing_info, ignore_index=True)
        data.index = range(data.shape[0])
        return data

    def Equal_Freqency(self,var, n):
        var = pd.DataFrame(var)
        var.columns = ['var']
        var['f'] = 1
        ### 剔除Null值
        # var = var[var['var'].isnull() == False]
        var_new = var.groupby(by='var').count()
        var_new['var'] = var_new.index
        var_new['F'] = var_new['f'].cumsum()
        f_avg = var.shape[0] / n
        bin = [-inf]
        for i in range(n):
            bin_cut = var_new.loc[abs((var_new['F'] - (i + 1) * f_avg)).idxmin(), 'var']
            if bin_cut not in bin:
                bin.append(bin_cut)
        bin = bin[:bin.__len__() - 1]
        bin.append(inf)
        var['bin'] = cut(var['var'], bin)
        # if len(var[var['bin'].isnull() == True]) > 0:
        #     var['bin'] = var['bin'].cat.add_categories(['Missing'])
        #     var['bin'].fillna('Missing', inplace=True)
        return var['bin']
    # 计算ks、auc、cut_off
    def ks_auc_evalute(self,y_pred=None,y_true=None):
        is_test = None
        if y_true is None or y_pred is None:
            is_test = 1
            return None,None,None
        if len(y_pred) != len(y_true):
            print('data length not match')
            return None,None,None
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred)
        val_auc = metrics.auc(fpr, tpr)
        imax = (tpr - fpr).argmax()
        val_ks = (tpr - fpr).max()
        cut_off = imax / len(tpr)
        return val_auc,val_ks,cut_off
    ## 绘制roc曲线
    def plot_roc(self, y_true=None, y_pred=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)
        ax.set_title('roc curve:  train')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid()
        ax.plot(fpr, tpr, 'r')
        ax.plot([0, 1], [0, 1], 'k--')
        # fig.show()
    # 绘制KS曲线
    def plot_ks(self, y_true=None, y_pred=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)
        n_sample = len(tpr)
        x = [i / n_sample for i in range(n_sample)]
        imax = (tpr - fpr).argmax()
        cut_off = imax / len(tpr)
        max_tpr = tpr[imax]
        max_fpr = fpr[imax]
        ax.set_title('K-S curve: train')
        ax.set_xlabel("Data Sets")
        ax.set_ylabel("Rate")
        ax.grid()
        ax.plot(x, tpr, 'r', label='True Positive Rate')
        ax.plot(x, fpr, 'b', label='False Positive Rate')
        ax.plot([cut_off, cut_off], [max_fpr, max_tpr], 'k--')
        # fig.show()
    # 单变量相关性系数计算
    def corr_calculate(self,x_val_1,x_val_2):
        df = pd.DataFrame({'x_var_1':x_val_1,'x_var_2':x_val_2})
        pearson = df.corr()
        spearman = df.corr('spearman')
        kendall = df.corr('kendall')
        return pearson['x_var_1']['x_var_2'],spearman['x_var_1']['x_var_2'],kendall['x_var_1']['x_var_2']
    # 计算woe和IV
    def woe_iv_calculate(self,data, x_var_name, y_var_name, bucket='Bucket'):
        if len(data[data[bucket].isnull() == True]) > 0:
            data[bucket] = data[bucket].cat.add_categories(['Missing'])
            data[bucket].fillna('Missing', inplace=True)
        df_ori = pd.DataFrame({x_var_name: data[x_var_name], "label": data[y_var_name], 'bucket': data[bucket]})
        df_group = df_ori.groupby('bucket', as_index=False)
        df_out = pd.DataFrame({'Bin_' + x_var_name: df_group.count().bucket, 'total': df_group.count().label,
                               'bad': df_group.sum().label, 'good': df_group.count().label - df_group.sum().label})
        df_out['odds'] = df_out['good'] / df_out['bad']
        df_out['bad_rate_bin'] = df_out['bad'] / (df_out['bad'] + df_out['good'])
        df_out['bad_rate_total'] = df_out['bad'] / (df_out['bad'].sum() + df_out['good'].sum())
        df_out['woe'] = np.log((df_out['bad'] / df_out['bad'].sum()) / (df_out['good'] / df_out['good'].sum()))
        df_out['IV'] = ((df_out['bad'] / df_out['bad'].sum()) - (df_out['good'] / df_out['good'].sum())) * df_out['woe']
        #plot to show bins with rate

        df_out.set_index('Bin_' + x_var_name,inplace=True)
        ax1 = df_out[["good", "bad"]].plot.bar(figsize=(10, 5))
        ax1.set_xticklabels(df_out.index, rotation=15)
        ax1.set_ylabel("good_bad_rate")
        ax1.set_title("good_bad_Bar")

        ax2 = df_out[["bad_rate_bin"]].plot(figsize=(10, 5))
        ax2.set_xticklabels(df_out.index, rotation=15)
        ax2.set_ylabel("bad_rate")
        ax2.set_title("bad_rata_bin")

        ax3 = df_out[["bad_rate_total"]].plot(figsize=(10, 5))
        ax3.set_xticklabels(df_out.index, rotation=15)
        ax3.set_ylabel("bad_rate")
        ax3.set_title("bad_rate_total")
        # plt.show()
        return df_out
    # 变量psi 计算
    def psi_calculate(self,data_train,x_var_name,x_test,train_bucket='Bucket'):
        psi_dict = {}
        x_val = data_train[x_var_name]
        y_val = data_train[train_bucket]
        df_ori_train = pd.DataFrame({'x_var': x_val, 'bucket': y_val})
        df_group_train = df_ori_train.groupby('bucket', as_index=False)
        df_bin_train = DataFrame({'Bin_' + x_var_name: df_group_train.count().bucket, 'cnt': df_group_train.count().x_var})
        bins = list(map(lambda x: x.right, df_bin_train['Bin_' + x_var_name]))
        bins = [-inf] + bins

        df_bin_train['Bin_' + x_var_name] = df_bin_train['Bin_' + x_var_name].astype(str)
        var_missing_train = x_val.iloc[where(x_val.isnull() == True)]
        if var_missing_train.shape[0] > 0:
            df_bin_train = self.MissingValue(x_val, df_bin_train)
        else:
            df_bin_train = df_bin_train

        df_bin_train['percent'] = df_bin_train['cnt'] / df_bin_train['cnt'].sum()

        df_ori_test = DataFrame({"x_var": x_test, "bucket": cut(x_test, bins=bins)})
        df_group_test = df_ori_test.groupby('bucket', as_index=False)
        df_bin_test = DataFrame({'Bin_' + x_var_name: df_group_test.count().bucket, 'cnt': df_group_test.count().x_var})

        var_missing_test = x_test.iloc[where(x_test.isnull() == True)]
        if var_missing_test.shape[0] > 0:
            df_bin_test = self.MissingValue(x_test, df_bin_test)
        else:
            df_bin_test = df_bin_test
        df_bin_test['percent'] = df_bin_test['cnt'] / df_bin_test['cnt'].sum()

        df_bin_train['percent_test'] = df_bin_test['percent']
        df_bin_train['ac-ex'] = df_bin_train['percent'] - df_bin_train['percent_test']
        df_bin_train['ln(ac/ex)'] = np.log(df_bin_train['percent'] / df_bin_train['percent_test'])
        df_bin_train['index'] = df_bin_train['ac-ex'] * df_bin_train['ln(ac/ex)']

        psi = df_bin_train['index'].sum()
        psi_dict[x_var_name] = psi
        # print(x_var_name, ':', psi)
        return psi_dict
    #多变量相关性系数计算
    def corr_calc_batch(self,data):
        pearson = data.corr()
        spearman = data.corr('spearman')
        kendall = data.corr('kendall')
        #绘制pearson相关系数图
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(3, 1, 1)
        xticks = list(pearson.index)  # x轴标签
        yticks = list(pearson.index)  # y轴标签
        sns.heatmap(pearson, annot=True, cmap="rainbow", ax=ax1, linewidths=.5,
                    annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
        ax1.set_xticklabels(xticks, rotation=35, fontsize=15)
        ax1.set_yticklabels(yticks, rotation=0, fontsize=15)
        ax1.set_title('pearson')
        #spearman相关系数图
        ax2 = fig.add_subplot(3, 1, 2)
        xticks = list(spearman.index)  # x轴标签
        yticks = list(spearman.index)  # y轴标签
        sns.heatmap(spearman, annot=True, cmap="rainbow", ax=ax2, linewidths=.5,
                    annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
        ax2.set_xticklabels(xticks, rotation=35, fontsize=15)
        ax2.set_yticklabels(yticks, rotation=0, fontsize=15)
        ax2.set_title('spearman')
        # #kendall相关系数
        ax3 = fig.add_subplot(3, 1, 3)
        xticks = list(kendall.index)  # x轴标签
        yticks = list(kendall.index)  # y轴标签
        sns.heatmap(kendall, annot=True, cmap="rainbow", ax=ax3, linewidths=.5,
                    annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
        ax3.set_xticklabels(xticks, rotation=35, fontsize=15)
        ax3.set_yticklabels(yticks, rotation=0, fontsize=15)
        ax3.set_title('kendall')
        fig.show()
        return pearson,spearman,kendall
