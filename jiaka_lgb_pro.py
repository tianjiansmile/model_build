import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import ks_2samp
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
import numpy as np
from Equal_frequency  import *
import math
import datetime

random_state = 8888
# sample_type = 'EI_PSI'
sample_type = 'Three'
# sample_type = 'EI'
model_file = 'model/tele_xy_three_{}.lgb'.format(sample_type)
importance_file = 'model/importance_tele_xy_three_{}.txt'.format(sample_type)
feature_name_column_file = 'model/feature_columns_tele_xy_three_{}.txt'.format(sample_type)
feature_score_file = 'model/feature_score_tele_xy_three_{}.txt'.format(sample_type)

three_col = ['cash_loan_15d', 'auth_contactnum_ratio_90d', 'auth_intimatenum_ratio_30d', 'org_count', 'auth_intimate_indirectnum_ratio_180d', 'datacoverge_90d', 'black_intimate_indirectnum_ratio_180d', 'match_score', 'black_indirect_peernum_ratio_90d', 'black_indirectnum_ratio_180d', 'black_intimate_indirect_peernum_ratio_30d', 'black_indirectnum_ratio_30d', 'other_count', 'auth_indirectnum_ratio_30d', 'cash_loan_30d', 'cash_loan_90d', 'consumstage_180d', 'blacklist_record_overdue_count', 'gray_record_overdue_count', 'creditScore', 'deviceRank', 'devicePrice', 'cityRec', 'provinceRec', 'countryRec', 'cityFreq', 'provinceFreq', 'countryFreq', 'ip90d', 'launchDay', 'appStability7d', 'appStability90d', 'appStability180d', 'top7d', 'top90d', 'top180d', 'tail7d', 'tail90d', 'tail180d', 'loan90d', 'loan7d', 'loan180d', 'finance7d', 'finance90d', 'finance180d', 'health7d', 'health90d', 'health180d', 'entertainment7d', 'entertainment90d', 'entertainment180d', 'tools7d', 'tools90d', 'tools180d', 'property7d', 'property90d', 'property180d', 'education7d', 'education90d', 'education180d', 'travel7d', 'travel90d', 'travel180d', 'woman7d', 'woman90d', 'woman180d', 'car7d', 'car90d', 'car180d', 'game7d', 'game90d', 'game180d', 'service7d', 'service90d', 'service180d', 'sns7d', 'sns90d', 'sns180d', 'shopping7d', 'shopping90d', 'shopping180d', 'reading7d', 'reading90d', 'reading180d', 'loan_account', 'loan_amt', 'outstand_count', 'loan_bal', 'overdue_count', 'overdue_amt', 'overdue_more_count', 'overdue_more_amt', 'generation_count', 'generation_amount', 'houmou_score', 'rnn']



def lgb_model(X_train, X_valid, y_train, y_valid):
    # np.random.seed(200)
    # np.random.shuffle(X_train)
    # np.random.seed(200)
    # np.random.shuffle(y_train)
    #

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
        'num_threads': 15}
    model = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=80000,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=4000,
                    verbose_eval=500)

    model.save_model(model_file, num_iteration=model.best_iteration)
    return(model)

def getAUC(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):

    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_auc = roc_auc_score(y_train.astype("int"), data_predict_1train)
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_auc = roc_auc_score(y_test.astype("int"), data_predict_1test)
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdueday.astype("int"), data_predict_1time)
    print("训练上的auc：%f \n验证集上的auc：%f \n测试集上的auc：%f "%(train_auc, test_auc, time_auc))

    return train_auc, test_auc, time_auc

def get_OOT_AUC(model, model_column,dfTime_X):

    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdueday.astype("int"), data_predict_1time)
    print("oot 上的auc：%f "%(time_auc))

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

def label_map(x):
    if int(x) >= 10:
        return 1
    else:
        return 0

def label_map_pd0(x):
    if int(x) > 0:
        return 1
    else:
        return 0

def train(trainData, testData, col):

    X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(np.array(trainData[col]),
                                                                          trainData['overdueday'].values,
                                                                          test_size=0.115,
                                                                          random_state=random_state)

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

    feat_imp = feature_importance(model,col)
    print(feat_imp[:50])

    feat_imp = pd.DataFrame(
        dict(columns=col, feature_importances=model.feature_importance() / model.feature_importance().sum()))
    feat_imp2 = feat_imp.sort_values(by=["feature_importances"], axis=0, ascending=False)
    feat_imp2[['columns', 'feature_importances']].to_csv(importance_file, sep=',', index=False, encoding='utf-8')
    
    print('写入评价结果')
    with open(feature_score_file, 'w')as wf:
        wf.write('train_auc: ' + str(train_auc) + 'val_auc: ' + str(test_auc) + 'time_auc：' + str(time_auc) + '\n')
        wf.write('train_ks: ' + str(train_ks) + 'val_ks: ' + str(test_ks) + 'time_ks：' + str(time_ks))

    return model


def oot_pred(model,col,oot_df):
    get_OOT_AUC(model,col,oot_df)
    compute_oot__ks(model,col,oot_df)


def ks_cal(label, pred):
    fpr, tpr, thresholds = roc_curve(label, pred)
    ks = max(tpr - fpr)
    return ks

def extend_clean(extend_df):
    print(extend_df.shape)
    num_features = [cont for cont in list(extend_df.select_dtypes(
        include=['float64', 'int64']).columns)]
    # print(num_features)
    # print(len(num_features))

    # 非数字类特征盘点
    txt_features = [cont for cont in list(extend_df.select_dtypes(
        exclude=['float64', 'int64']).columns) if cont not in ['EI_score_友盟分','EI_nifadata_loanamt']]

    # print(txt_features)
    print('txt len',len(txt_features))

    return extend_df[num_features]

def data_split(allData,product_id):
    allData['split1'] = allData['product_id'].apply(
        lambda x: 'oot' if x == product_id
                           # and x <= '2019-05-01'
        else 'train')
    for name, group in allData.groupby('split1'):
        print(name, group.shape)
        if name is 'oot':
            oot1 = group
        else:
            train = group

    oot1.drop(['split1'], axis=1, inplace=True)

    return oot1

def oot_train_split(allData):
    create_time = allData['create_time']
    create_time = create_time.sort_values()
    create_time.index = range(len(create_time))
    oot_index = round(len(create_time) * 0.9)
    oot_time = create_time[oot_index]

    allData = allData.assign(data_set=allData['create_time'].apply(lambda x: 'train' \
        if datetime.datetime.strptime(x[0:10], "%Y-%m-%d") <= datetime.datetime.strptime(oot_time, "%Y-%m-%d") \
        else 'test'))

    # groupby = allData.groupby(allData['split'])
    # for name, group in allData.groupby('split'):
    #     print(name,group.shape)
    #     if name is 'oot':
    #         oot = group
    #     else:
    #         train = group

    oot_df = allData[allData['data_set'] == 'test']
    train_df = allData[allData['data_set'] == 'train']

    return train_df,oot_df

def oot_train_split_pro(allData):
    allData['split1'] = allData['create_time'].apply(
        lambda x: 'oot' if x >= '2019-03-15' and x <= '2019-03-31' else 'train')
    for name, group in allData.groupby('split1'):
        print(name,group.shape)
        if name is 'oot':
            oot1 = group
        else:
            train = group

    allData['split2'] = allData['create_time'].apply(
        lambda x: 'oot' if x >= '2019-04-01' and x <= '2019-04-30' else 'train')
    for name, group in allData.groupby('split2'):
        print(name, group.shape)
        if name is 'oot':
            oot2 = group
        else:
            train = group

    allData['split3'] = allData['create_time'].apply(
        lambda x: 'oot' if x >= '2019-05-01' else 'train')
    for name, group in allData.groupby('split3'):
        print(name, group.shape)
        if name is 'oot':
            oot3 = group
        else:
            train = group

    return train,oot1,oot2,oot3


def mix_feature1(allData):

    # allData = pd.read_csv('all_feature.csv')


    print(allData.shape)

    # 切分数据集
    allData = data_split(allData)

    feature_jinpan_drop(allData)
    # feature_three_drop(allData)

    print('all', allData.shape)

    print(allData['Unnamed: 0.1.1'].unique())


    # 确保二分类
    allData['overdueday'] = allData['label']
    allData['houmou_score'] = allData['houmou_score_x']


    allData,oot_df = oot_train_split(allData)

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'overdue_days', 'Unnamed: 0.1.1','Unnamed: 0','Unnamed: 0.1',
                  'score','obscure_score','label','houmou_score_y','houmou_score_x'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'overdue_days', 'label','Unnamed: 0.1.1','Unnamed: 0','Unnamed: 0.1',
                 'score','obscure_score','houmou_score_y','houmou_score_x'], axis=1, inplace=True)


    psi_extend_col = [
        'EI_sar_contact_list_cnt_needsType_骚扰电话', 'EI_sar_contact_list_ratioOfCnt_needsType_未知',
        'EI_sar_contact_list_ratioOfCnt_needsType_骚扰电话','EI_sar_main_service_ratio_company_type_通信服务机构']

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if
                    'EI' not in cont and
                    # 'contact_list_ratioOfCnt' not in cont and
                    'mb_infos_lastweek' not in cont and
                    'rulesDetail' not in cont and
                    'tongdunguarddata' not in cont and
                    'tanzhidata_platform_Infos' not in cont and
                    #
                    'live_addr' not in cont and
                    'card_addr' not in cont and
                    cont not in psi_extend_col and
                    'overdueday' != cont]

    # print(cat_features)
    print('入模 ',len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    # print('psi_col',psi_col)
    #
    #
    # cat_features = [cont for cont in cat_features if cont not in psi_col]
    # print('PSI后',cat_features)
    # print(len(cat_features))

    # his_count = allData['apply_sum_all'].count()
    # call_max_overdue_sum_count = allData['call_max_overdue_sum'].count()


    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)
    # 训练
    model = train(allData, oot_df, cat_features)

    data = {}
    writerCSV = pd.DataFrame(columns=cat_features, data=data)

    writerCSV.to_csv(feature_name_column_file, sep=',', encoding='utf-8', index=False)

def mix_feature2(allData):

    # allData = pd.read_csv('all_feature.csv')

    global  sample_type
    sample_type = 'EI'

    print(allData.shape)

    # 切分数据集
    allData = data_split(allData)

    feature_jinpan_drop(allData)
    # feature_three_drop(allData)

    print('all', allData.shape)

    print(allData['Unnamed: 0.1.1'].unique())


    # 确保二分类
    allData['overdueday'] = allData['label']
    allData['houmou_score'] = allData['houmou_score_x']


    allData,oot_df = oot_train_split(allData)

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'overdue_days', 'Unnamed: 0.1.1','Unnamed: 0','Unnamed: 0.1',
                  'score','obscure_score','label','houmou_score_y','houmou_score_x'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'overdue_days', 'label','Unnamed: 0.1.1','Unnamed: 0','Unnamed: 0.1',
                 'score','obscure_score','houmou_score_y','houmou_score_x'], axis=1, inplace=True)


    psi_extend_col = [
        'EI_sar_contact_list_cnt_needsType_骚扰电话', 'EI_sar_contact_list_ratioOfCnt_needsType_未知',
        'EI_sar_contact_list_ratioOfCnt_needsType_骚扰电话','EI_sar_main_service_ratio_company_type_通信服务机构']

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if
                    # 'EI' not in cont and
                    # 'contact_list_ratioOfCnt' not in cont and
                    'mb_infos_lastweek' not in cont and
                    'rulesDetail' not in cont and
                    'tongdunguarddata' not in cont and
                    'tanzhidata_platform_Infos' not in cont and
                    #
                    'live_addr' not in cont and
                    'card_addr' not in cont and
                    cont not in psi_extend_col and
                    'overdueday' != cont]

    # print(cat_features)
    print('入模 ',len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    # print('psi_col',psi_col)
    #


    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)
    # 训练
    model = train(allData, oot_df, cat_features)

    data = {}
    writerCSV = pd.DataFrame(columns=cat_features, data=data)

    writerCSV.to_csv(feature_name_column_file, sep=',', encoding='utf-8', index=False)


def mix_feature3(allData):
    # allData = pd.read_csv('all_feature.csv')

    global sample_type
    sample_type = 'jinpan'

    print(allData.shape)

    # 切分数据集
    allData = data_split(allData)

    # feature_jinpan_drop(allData)
    # feature_three_drop(allData)

    print('all', allData.shape)

    print(allData['Unnamed: 0.1.1'].unique())

    # 确保二分类
    allData['overdueday'] = allData['label']
    allData['houmou_score'] = allData['houmou_score_x']

    allData, oot_df = oot_train_split(allData)

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'overdue_days', 'Unnamed: 0.1.1', 'Unnamed: 0', 'Unnamed: 0.1',
                  'score', 'obscure_score', 'label', 'houmou_score_y', 'houmou_score_x'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'overdue_days', 'label', 'Unnamed: 0.1.1', 'Unnamed: 0', 'Unnamed: 0.1',
                 'score', 'obscure_score', 'houmou_score_y', 'houmou_score_x'], axis=1, inplace=True)

    psi_extend_col = [
        'EI_sar_contact_list_cnt_needsType_骚扰电话', 'EI_sar_contact_list_ratioOfCnt_needsType_未知',
        'EI_sar_contact_list_ratioOfCnt_needsType_骚扰电话', 'EI_sar_main_service_ratio_company_type_通信服务机构']

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if
                    # 'EI' not in cont and
                    # 'contact_list_ratioOfCnt' not in cont and
                    'mb_infos_lastweek' not in cont and
                    'rulesDetail' not in cont and
                    'tongdunguarddata' not in cont and
                    'tanzhidata_platform_Infos' not in cont and
                    #
                    'live_addr' not in cont and
                    'card_addr' not in cont and
                    cont not in psi_extend_col and
                    'overdueday' != cont]

    # print(cat_features)
    print('入模 ', len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    # print('psi_col',psi_col)
    #

    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)
    # 训练
    model = train(allData, oot_df, cat_features)

    data = {}
    writerCSV = pd.DataFrame(columns=cat_features, data=data)

    writerCSV.to_csv(feature_name_column_file, sep=',', encoding='utf-8', index=False)

def old_tele_feature():

    allData = pd.read_csv('tele_xy_three_feature_.csv')

    print('all', allData.shape)

    # 切分数据集
    allData = data_split(allData)

    print('all', allData.shape)

    # 确保二分类
    allData['overdueday'] = allData['label']

    allData,oot_df = oot_train_split(allData)

    print('train', allData.shape)
    print('oot_df', oot_df.shape)
    #
    # # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time','label', 'overdue_days','score','obscure_score','Unnamed: 0'], axis=1, inplace=True)

    # # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'label', 'overdue_days','score','obscure_score','Unnamed: 0'], axis=1, inplace=True)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if cont != 'overdueday']

    # psi_col = calc_PSI(allData, oot_df, cat_features)
    #
    # # 不去除第三方特征
    # psi_col = [cont for cont in psi_col if cont not in three_col]
    # print('psi_col', len(psi_col))
    #
    # cat_features = [cont for cont in cat_features if cont not in psi_col]

    print('入模特征',len(cat_features))

    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)
    # 训练
    model = train(allData, oot_df, cat_features)

def calc_PSI(allData,oot_df,cat_features):
    # print(allData.shape)
    # print(oot_df.shape)

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

# 删除网络特征
def feature_call_drop(train):
    col_drop = ['call_max_overdue_sum','call_max_overdue_pdl','call_max_overdue_int',
                'call_max_overdue_times','call_max_apply_sum','call_max_approve_sum',
                'call_max_loanamount_sum','call_avg_overdue_sum','call_avg_overdue_pdl',
                'call_avg_overdue_int','call_avg_overdue_times','call_avg_apply_sum',
                'call_avg_approve_sum','call_avg_loanamount_sum']

    train.drop(col_drop, axis=1, inplace=True)

def feature_contact_drop(train):
    col_drop = ['contact_max_overdue_sum','contact_max_overdue_pdl','contact_max_overdue_int',
                'contact_max_overdue_times','contact_max_apply_sum','contact_max_approve_sum',
                'contact_max_loanamount_sum','contact_avg_overdue_sum','contact_avg_overdue_pdl',
                'contact_avg_overdue_int','contact_avg_overdue_times','contact_avg_apply_sum',
                'contact_avg_approve_sum','contact_avg_loanamount_sum']

    train.drop(col_drop, axis=1, inplace=True)

def feature_three_drop(allData):
    col_drop = ['cash_loan_15d', 'auth_contactnum_ratio_90d', 'auth_intimatenum_ratio_30d', 'org_count', 'auth_intimate_indirectnum_ratio_180d', 'datacoverge_90d', 'black_intimate_indirectnum_ratio_180d', 'match_score', 'black_indirect_peernum_ratio_90d', 'black_indirectnum_ratio_180d', 'black_intimate_indirect_peernum_ratio_30d', 'black_indirectnum_ratio_30d', 'other_count', 'auth_indirectnum_ratio_30d', 'cash_loan_30d', 'cash_loan_90d', 'consumstage_180d', 'blacklist_record_overdue_count', 'gray_record_overdue_count', 'creditScore', 'deviceRank', 'devicePrice', 'cityRec', 'provinceRec', 'countryRec', 'cityFreq', 'provinceFreq', 'countryFreq', 'ip90d', 'launchDay', 'appStability7d', 'appStability90d', 'appStability180d', 'top7d', 'top90d', 'top180d', 'tail7d', 'tail90d', 'tail180d', 'loan90d', 'loan7d', 'loan180d', 'finance7d', 'finance90d', 'finance180d', 'health7d', 'health90d', 'health180d', 'entertainment7d', 'entertainment90d', 'entertainment180d', 'tools7d', 'tools90d', 'tools180d', 'property7d', 'property90d', 'property180d', 'education7d', 'education90d', 'education180d', 'travel7d', 'travel90d', 'travel180d', 'woman7d', 'woman90d', 'woman180d', 'car7d', 'car90d', 'car180d', 'game7d', 'game90d', 'game180d', 'service7d', 'service90d', 'service180d', 'sns7d', 'sns90d', 'sns180d', 'shopping7d', 'shopping90d', 'shopping180d', 'reading7d', 'reading90d', 'reading180d', 'loan_account', 'loan_amt', 'outstand_count', 'loan_bal', 'overdue_count', 'overdue_amt', 'overdue_more_count', 'overdue_more_amt', 'generation_count', 'generation_amount', 'houmou_score', 'rnn']

    allData.drop(col_drop, axis=1, inplace=True)

import jinpan_col
def feature_jinpan_drop(train):
    col_drop = jinpan_col.col_drop
    train.drop(col_drop, axis=1, inplace=True)

def MissingValue(var,d3):
    var_missing = var.iloc[where(var.isnull()==True)]

    missing_percent = var_missing.shape[0]/var.shape[0]

    missing_total_num = var_missing.shape[0]

    missing_info = []
    missing_info.append('missing value')
    missing_info.append(missing_total_num)

    missing_info = DataFrame([missing_info],columns=d3.columns)
    d3 = d3.append(missing_info,ignore_index=True)
    d3.index = range(d3.shape[0])
    return d3

def extend_feature():
    allData = pd.read_csv('extend_info_feature.csv')

    # 横向合并数据集
    # allData = pd.concat([addrData,allData],axis=1)

    print(allData.shape)

    # 暂时删除有空数据的行
    # allData.dropna(axis=0, how='any', inplace=True)

    # 确保二分类
    allData['overdueday'] = allData['label']

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'label', 'Unnamed: 0',
                  'EI_sar_JSON_INFO_lastmth0_cell_phone_num'], axis=1, inplace=True)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdueday']

    print(cat_features)

    trainData, testData = train_test_split(allData, test_size=0.2, random_state=43)

    train_rate = trainData.overdueday.sum() / trainData.shape[0]
    test_rate = testData.overdueday.sum() / testData.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)

    # 训练
    train(trainData, testData, cat_features)

def jinpan_concat():
    jinpan_2 = pd.read_csv('data/jiaka_jinpan_feature_range(0, 20000).txt')
    jinpan_4 = pd.read_csv('data/jiaka_jinpan_feature_range(20000, 40000).txt')
    jinpan_6 = pd.read_csv('data/jiaka_jinpan_feature_range(40000, 60000).txt')
    jinpan_8 = pd.read_csv('data/jiaka_jinpan_feature_range(60000, 80000).txt')

    frames = [jinpan_2, jinpan_4,jinpan_6,jinpan_8]

    return pd.concat(frames)

def jinpan_feature():
    # 纵向合并
    allData = jinpan_concat()

    print(allData.shape)

    # 暂时删除有空数据的行
    # allData.dropna(axis=0, how='any', inplace=True)

    # 确保二分类
    allData['overdueday'] = allData['label']

    # 将不参与训练的特征数据删除
    allData.drop(['order_id','create_time','loan_no','overdue_days','label'], axis=1, inplace=True)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdueday']

    print(cat_features)

    trainData, testData = train_test_split(allData, test_size=0.2, random_state=43)

    train_rate = trainData.overdueday.sum() / trainData.shape[0]
    test_rate = testData.overdueday.sum() / testData.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)

    # 训练
    train(trainData, testData, cat_features)


def data_concat(allData):
    # allData = pd.read_csv('all_feature.csv')

    print(allData.shape)

    # 切分数据集
    allData = data_split(allData)

    # feature_jinpan_drop(allData)
    # feature_three_drop(allData)

    print('all', allData.shape)

    print(allData['Unnamed: 0.1.1'].unique())

    allData.to_csv('all_feature_split.csv')

if __name__ == '__main__':
    import time

    starttime = time.time()

    allData = pd.read_csv('all_feature.csv')

    # 评估混合特征
    # mix_feature1(allData)
    #
    # mix_feature2(allData)
    # 
    # mix_feature3(allData)

    data_concat(allData)

    # 原特征
    # old_tele_feature()

    # mix_all_feature()

    # extend_feature()

    # jinpan_feature()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)