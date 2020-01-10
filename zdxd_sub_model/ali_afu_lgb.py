import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import ks_2samp
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import model_selection
import numpy as np
from Equal_frequency import *
import math

random_state = 8888
# sample_type = 'EI_PSI'
sample_type = 'afu'
# sample_type = 'EI'

model_file = 'model/model_{}.lgb'.format(sample_type)
importance_file = 'model/importance_{}.txt'.format(sample_type)
feature_name_column_file = 'model/feature_columns_{}.txt'.format(sample_type)
feature_score_file = 'model/feature_score_{}.txt'.format(sample_type)


def lgb_model(X_train, X_valid, y_train, y_valid):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.001,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 10,
        'verbose': -1,
        # 'max_depth':4,
        'num_threads': 25}
    model = lgb.train(params=params,
                      train_set=lgb_train,
                      num_boost_round=80000,
                      valid_sets=[lgb_train, lgb_eval],
                      early_stopping_rounds=2000,
                      verbose_eval=500)

    model.save_model(model_file, num_iteration=model.best_iteration)
    return (model)


def getAUC(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):
    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_auc = roc_auc_score(y_train.astype("int"), data_predict_1train)
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_auc = roc_auc_score(y_test.astype("int"), data_predict_1test)
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdueday.astype("int"), data_predict_1time)
    print("训练上的auc：%f \n验证集上的auc：%f \n测试集上的auc：%f " % (train_auc, test_auc, time_auc))
    return train_auc, test_auc, time_auc

def get_OOT_AUC(model, model_column, dfTime_X):
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdueday.astype("int"), data_predict_1time)
    print("oot 上的auc：%f " % (time_auc))


def compute_oot__ks(model, model_column, dfTime_X):
    get_ks = lambda proba, target: ks_2samp(proba[target == 1], proba[target != 1]).statistic
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_ks = get_ks(data_predict_1time, dfTime_X.overdueday.astype("int"))

    print("oot 上的KS：%f " % (time_ks))


def compute_ks(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):
    '''
    target: numpy array of shape (1,)
    proba: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation

    '''
    get_ks = lambda proba, target: ks_2samp(proba[target == 1], proba[target != 1]).statistic
    target = dfTime_X.overdueday.astype("int")
    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_ks = get_ks(data_predict_1train, y_train.astype("int"))
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_ks = get_ks(data_predict_1test, y_test.astype("int"))
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_ks = get_ks(data_predict_1time, dfTime_X.overdueday.astype("int"))

    print("训练上的KS：%f \n测试上的KS：%f \n跨时间上的KS：%f " % (train_ks, test_ks, time_ks))
    return train_ks, test_ks, time_ks

def feature_importance(model, column_name):
    """get_top_10_feature_importance
       param model:model
    """
    feat_imp = pd.DataFrame(
        dict(columns=column_name, feature_importances=model.feature_importance() / model.feature_importance().sum()))
    feat_imp2 = feat_imp.sort_values(by=["feature_importances"], axis=0, ascending=False).head(100)
    return (feat_imp2)


def label_map_pd0(x):
    if int(x) > 0:
        return 1
    else:
        return 0


def train(trainData, testData, col):
    X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(np.array(trainData[col]),
                                                                          trainData['overdueday'].values,
                                                                          test_size=0.10,
                                                                          random_state=123)

    model = lgb_model(X_train, X_valid, Y_train, Y_valid)

    train_auc, test_auc, time_auc = getAUC(model,
           col,
           X_train,
           Y_train,
           X_valid,
           Y_valid,
           testData)

    train_ks, test_ks, time_ks = compute_ks(model,
               col,
               X_train,
               Y_train,
               X_valid,
               Y_valid,
               testData)

    fea_impt = feature_importance(model, col)
    print(fea_impt[:30])

    feat_imp = pd.DataFrame(
        dict(columns=col, feature_importances=model.feature_importance() / model.feature_importance().sum()))
    feat_imp2 = feat_imp.sort_values(by=["feature_importances"], axis=0, ascending=False)
    feat_imp2[['columns', 'feature_importances']].to_csv(importance_file, sep=',', index=False, encoding='utf-8')

    print('写入评价结果')
    with open(feature_score_file, 'w')as wf:
        wf.write('train_auc: ' + str(train_auc) + 'val_auc: ' + str(test_auc) + 'time_auc：' + str(time_auc) + '\n')
        wf.write('train_ks: ' + str(train_ks) + 'val_ks: ' + str(test_ks) + 'time_ks：' + str(time_ks))

    return model


def oot_pred(model, col, oot_df):
    get_OOT_AUC(model, col, oot_df)
    compute_oot__ks(model, col, oot_df)


def oot_train_split_pro(allData):
    allData['split'] = allData['create_time'].apply(lambda x: 'oot' if str(x) >= '20190601' and str(x) <= '20190610' else 'train')
    groupby = allData.groupby(allData['split'])
    for name, group in allData.groupby('split'):
        if name is 'oot':
            oot = group
        else:
            train = group

    oot.drop(['split'], axis=1, inplace=True)
    train.drop(['split'], axis=1, inplace=True)

    return train,oot


def oot_train_split(allData):
    create_time = allData['create_time']
    create_time = create_time.sort_values()
    create_time.index = range(len(create_time))
    oot_index = round(len(create_time) * 0.9)
    oot_time = create_time[oot_index]

    allData = allData.assign(data_set=allData['create_time'].apply(lambda x: 'train' \
        if datetime.strptime(str(x), "%Y%m%d") <= datetime.strptime(str(oot_time), "%Y%m%d") \
        else 'test'))

    oot_df = allData[allData['data_set'] == 'test']
    train_df = allData[allData['data_set'] == 'train']

    return train_df, oot_df


def label_map(x):
    if int(x) >= 7:
        return 1
    else:
        return 0


def data_split(allData, product_id):
    allData['split1'] = allData['product_id'].apply(
        lambda x: 'oot' if x == product_id else 'train')
    for name, group in allData.groupby('split1'):
        if name is 'oot':
            oot1 = group

    oot1.drop(['split1'], axis=1, inplace=True)
    return oot1




def afu_feature_pro():
    allData = pd.read_excel('data/zcaf_data.xlsx', encoding='utf8')
    # allData = pd.read_csv('jiaka_social_fea.csv')

    print(allData.shape)

    # 暂时删除有空数据的行
    # allData.dropna(axis=0, how='any', inplace=True)
    # indexs = list(allData[np.isnan(allData['SN_one_step_degree'])].index)
    # allData = allData.drop(indexs)
    # print(allData['product_id'].unique().shape)
    # len(data)

    # allData['label'] = allData['overdue_days'].map(label_map)
    # 确保二分类
    allData['overdueday'] = allData['label']
    # allData['overdueday'] = allData['m1_enev']

    allData, oot_df = oot_train_split_pro(allData)

    # allData, oot_df = oot_train_split(allData)

    print(allData['create_time'].unique(), oot_df['create_time'].unique())

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'md5_mobile', 'name', 'overdue_days',
                  'product_name', 'loan','身份证','手机号','label'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'md5_mobile', 'name', 'overdue_days',
                  'product_name', 'loan','身份证','手机号','label'], axis=1, inplace=True)

    # 去除extend_info 特征
    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    # 'high' not in cont and

                    cont != 'overdueday']

    cat_features = ['阿福分','欺诈评分']

    # cat_features = [cont for cont in list(allData.select_dtypes(
    #     exclude=['object']).columns) if
    #                 # 'high' not in cont and
    #
    #                 cont != 'overdueday']

    # cat_features.append('阿福分')

    print(cat_features)
    print(len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    print('psi_col', psi_col)

    # cat_features = [cont for cont in cat_features if cont not in psi_col]
    # print('PSI后',cat_features)
    # print(len(cat_features))

    # his_count = allData['SN_one_step_age_std'].count()
    # call_max_overdue_sum_count = allData['SN_one_step_apply_mean'].count()

    # print('SN_two_step_age_std',his_count,his_count/allData.shape[0],'SN_one_step_apply_mean',call_max_overdue_sum_count,call_max_overdue_sum_count/allData.shape[0])

    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)

    # 训练
    # 训练
    model = train(allData, oot_df, cat_features)

    data = {}
    writerCSV = pd.DataFrame(columns=cat_features, data=data)

    writerCSV.to_csv(feature_name_column_file, sep=',', encoding='utf-8', index=False)






def calc_PSI(allData, oot_df, cat_features):
    print(allData.shape)
    print(oot_df.shape)

    psi_col = []

    n = 0
    for col in cat_features:
        n += 1
        # if n == 4:
        #     break
        v1 = allData[col]
        v2 = oot_df[col]

        try:
            d1 = DataFrame({"Var": v1, "Bucket": Equal_Freqency(v1, 10)})
            d2 = d1.groupby('Bucket', as_index=False)
            d3 = DataFrame({'Bin_' + col: d2.count().Bucket, 'cnt': d2.count().Var})

            bins = list(map(lambda x: x.right, d3['Bin_' + col]))
            bins = [-inf] + bins

            d3['Bin_' + col] = d3['Bin_' + col].astype(str)

            var_missing_1 = v1.iloc[where(v1.isnull() == True)]
            if var_missing_1.shape[0] > 0:
                d3_v1 = MissingValue(v1, d3)
            else:
                d3_v1 = d3
            d3_v1['percent'] = d3_v1['cnt'] / d3_v1['cnt'].sum()

            d1 = DataFrame({"Var": v2, "Bucket": cut(v2, bins=bins)})
            d2 = d1.groupby('Bucket', as_index=False)
            d3 = DataFrame({'Bin_' + col: d2.count().Bucket, 'cnt': d2.count().Var})

            var_missing_2 = v2.iloc[where(v2.isnull() == True)]
            if var_missing_2.shape[0] > 0:
                d3_v2 = MissingValue(v2, d3)
            else:
                d3_v2 = d3
            d3_v2['percent'] = d3_v2['cnt'] / d3_v2['cnt'].sum()

            d3_v1['percent_2'] = d3_v2['percent']
            d3_v1['ac-ex'] = d3_v1['percent'] - d3_v1['percent_2']
            d3_v1['ln(ac/ex)'] = log(d3_v1['percent'] / d3_v1['percent_2'])
            d3_v1['index'] = d3_v1['ac-ex'] * d3_v1['ln(ac/ex)']
            # print(d3_v1)
            psi = d3_v1['index'].sum()

            if psi >= 0.3:
                # print(col)
                # print(psi)
                if psi != float('Inf'):
                    print(col)
                    print(psi)
                    psi_col.append(col)
        except:
            continue

    return psi_col


def MissingValue(var, d3):
    var_missing = var.iloc[where(var.isnull() == True)]

    missing_percent = var_missing.shape[0] / var.shape[0]

    missing_total_num = var_missing.shape[0]

    missing_info = []
    missing_info.append('missing value')
    missing_info.append(missing_total_num)

    missing_info = DataFrame([missing_info], columns=d3.columns)
    d3 = d3.append(missing_info, ignore_index=True)
    d3.index = range(d3.shape[0])
    return d3


if __name__ == '__main__':
    import time

    starttime = time.time()

    afu_feature_pro()
    endtime = time.time()
    print(' cost time: ', endtime - starttime)
