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
from Equal_frequency  import *
import math


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
        'num_threads': 15}
    model = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=80000,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=2000,
                    verbose_eval=500)
    #model.save_model('/suanhua/model/'+model_name+'.model')
    return(model)

def getAUC(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):

    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_auc = roc_auc_score(y_train.astype("int"), data_predict_1train)
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_auc = roc_auc_score(y_test.astype("int"), data_predict_1test)
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdueday.astype("int"), data_predict_1time)
    print("训练上的auc：%f \n验证集上的auc：%f \n测试集上的auc：%f "%(train_auc, test_auc, time_auc))

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
                                                                          test_size=0.10,
                                                                          random_state=123)

    model = lgb_model(X_train, X_valid, Y_train, Y_valid)

    getAUC(model,
           col,
           X_train,
           Y_train,
           X_valid,
           Y_valid,
           testData)

    compute_ks(model,
               col,
               X_train,
               Y_train,
               X_valid,
               Y_valid,
               testData)

    fea_impt = feature_importance(model, col)[:100]
    print(fea_impt)

    return model


def oot_pred(model,col,oot_df):
    get_OOT_AUC(model,col,oot_df)
    compute_oot__ks(model,col,oot_df)



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

def oot_train_split(allData):
    allData['split'] = allData['create_time'].apply(lambda x: 'oot' if x >= '2019-03' else 'train')
    groupby = allData.groupby(allData['split'])
    for name, group in allData.groupby('split'):
        print(name,group.shape)
        if name is 'oot':
            oot = group
        else:
            train = group

    return train,oot

def mix_feature():

    allData = pd.read_csv('data/social_emer_2019_test1.csv')
    #allData = allData.head(180000)
    #allData = pd.read_csv('data/social_emer_2019_6w_test3.csv')
    #allData = pd.read_csv('data/social_comm_feature_all.csv')

    #allData =  data_split(allData,12)
    #allData = pd.read_csv('jiaka_social_fea.csv')


    print(allData.shape)

    # 暂时删除有空数据的行
    #allData.dropna(axis=0, how='any', inplace=True)
    #indexs = list(allData[np.isnan(allData['SN_one_step_degree'])].index)
   # allData = allData.drop(indexs)
    print(allData['product_id'].unique().shape)
    #len(data)
    #allData.drop_duplicates(subset=['md5_num'],keep='first')
    #print(allData.shape)
    #allData = allData[~allData['related_user_count'].isin([0])]
    #allData = allData[~allData['SN_one_step_degree'].isin([2])]
    #allData = allData[~allData['SN_one_step_degree'].isin([3])]
    #allData = allData[~allData['SN_one_step_degree'].isin([4])]
    #allData = allData[~allData['SN_one_step_degree'].isin([5])]
    #allData = allData[~allData['SN_one_step_degree'].isin([6])]
    #allData = allData[~allData['SN_one_step_degree'].isin([7])]

    #allData['label'] = allData['overdue_days'].map(label_map)
    # 确保二分类
    #allData['overdueday'] = allData['label']
    allData['overdueday'] = allData['m1_enev']

    #allData = oot_train_split_pro(allData)

    allData,oot_df = oot_train_split(allData)

    print(allData['create_time'].unique(),oot_df['create_time'].unique())

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'md5_num', 'm1_enev', 'first_overdue_days',
        'product_id','slpd7','m1_times','m2','m3','m4','m5'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'md5_num', 'm1_enev', 'first_overdue_days',
                'product_id','slpd7','m1_times','m2','m3','m4','m5'], axis=1, inplace=True)

# 将不参与训练的特征数据删除
    #allData.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label'], axis=1, inplace=True)

        # 将不参与训练的特征数据删除
    #oot_df.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label'], axis=1, inplace=True)



    # 去除extend_info 特征
    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    #'high' not in cont and
                    #'pdl' not in cont and
                    #'comm' not in cont and
                    #'two' not in cont and
                    'model_score' not in cont and
                    'apply' not in cont and
                    #'reject' not in cont and
		    #'overdue' not in cont and
		    #'middle' not in cont and
		    #'interval' not in cont and
                    'pdl' not in cont and
                    'int' not in cont and
                    'overdue' not in cont and
                    'loanamount' not in cont and
                    'maxOverdue' not in cont and
                    'approve' not in cont and
                    'comm' not in cont and
                    'pd7' not in cont and
                    'AO' not in cont and
                    #'apply' not in cont and
                    cont != 'overdueday']



    print(cat_features)
    print(len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    print('psi_col',psi_col)


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

def mix_feature():

    allData = pd.read_csv('all_feature_1.csv')


    print(allData.shape)

    # 暂时删除有空数据的行
    # allData.dropna(axis=0, how='any', inplace=True)

    # 确保二分类
    allData['overdueday'] = allData['label']


    # feature_call_drop(allData)
    # feature_contact_drop(allData)
    feature_jinpan_drop(allData)

    allData,oot_df = oot_train_split(allData)

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label', 'Unnamed: 0',
                  'md5_num', 'EI_sar_JSON_INFO_lastmth0_cell_phone_num', 'label'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label', 'Unnamed: 0',
                  'md5_num', 'EI_sar_JSON_INFO_lastmth0_cell_phone_num', 'label'], axis=1, inplace=True)


    # 查看该文本特征的分类
    # print(extend_df['EI_tanzhidata_mb_infos_lastweek5_overdue_average_level'].unique())
    # # 转换
    # extend_df['EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level'] = pd.factorize(
    #     extend_df["EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level"])[0].astype(np.uint16)
    #
    # print(extend_df['EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level'].unique())


    # cat_features = [cont for cont in list(allData.select_dtypes(
    #     include=['float64', 'int64']).columns) if cont != 'overdueday']

    # 去除extend_info 特征
    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    'EI' not in cont and
                    # 'live_addr' not in cont and
                    #  'card_addr' not in cont and
                    cont != 'overdueday']

    psi_extend_col = [
        'EI_sar_contact_list_cnt_needsType_骚扰电话', 'EI_sar_contact_list_ratioOfCnt_needsType_未知',
        'EI_sar_contact_list_ratioOfCnt_needsType_骚扰电话']

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if
                    # 'EI' in cont and
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

    print(cat_features)
    print(len(cat_features))

    psi_col = calc_PSI(allData, oot_df, cat_features)
    print('psi_col',psi_col)


    cat_features = [cont for cont in cat_features if cont not in psi_col]
    # print('PSI后',cat_features)
    print(len(cat_features))

    # his_count = allData['apply_sum_all'].count()
    # call_max_overdue_sum_count = allData['call_max_overdue_sum'].count()

    # print('训练jinpan命中情况',his_count,his_count/allData.shape[0],'网络特征命中情况',call_max_overdue_sum_count,call_max_overdue_sum_count/allData.shape[0])


    trainData, testData = train_test_split(allData, test_size=0.11,random_state=43)

    train_rate = trainData.overdueday.sum() / trainData.shape[0]
    test_rate = testData.overdueday.sum() / testData.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)
    # 训练
    model = train(trainData, testData, cat_features)

    oot_rate = oot_df.overdueday.sum() / oot_df.shape[0]
    print(' oot_rate: ', oot_rate)
    oot_pred(model,cat_features,oot_df)


def mix_all_feature():

    allData = pd.read_csv('data/all_feature_2.csv')


    print(allData.shape)

    # 暂时删除有空数据的行
    # allData.dropna(axis=0, how='any', inplace=True)

    # 确保二分类
    allData['overdueday'] = allData['label']


    # feature_call_drop(allData)
    # feature_contact_drop(allData)

    allData,oot_df = oot_train_split(allData)

    # 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label', 'Unnamed: 0',
                  'md5_num', 'EI_sar_JSON_INFO_lastmth0_cell_phone_num', 'label','score','obscure_score'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'loan_no', 'overdue_days', 'label', 'Unnamed: 0',
                  'md5_num', 'EI_sar_JSON_INFO_lastmth0_cell_phone_num', 'label','score','obscure_score'], axis=1, inplace=True)

    
    psi_extend_col = [
        'EI_sar_contact_list_cnt_needsType_骚扰电话', 'EI_sar_contact_list_ratioOfCnt_needsType_未知',
        'EI_sar_contact_list_ratioOfCnt_needsType_骚扰电话']
    # 去除extend_info 特征
    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    # 'EI' not in cont and
                   'tongdunguarddata' not in cont and
                   'card_addr' not in cont and
                    cont not in psi_extend_col and
                    cont != 'overdueday']


    print(cat_features)
    print(len(cat_features))

    trainData, testData = train_test_split(allData, test_size=0.11,random_state=43)

    train_rate = trainData.overdueday.sum() / trainData.shape[0]
    test_rate = testData.overdueday.sum() / testData.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)
    # 训练
    model = train(trainData, testData, cat_features)

    oot_rate = oot_df.overdueday.sum() / oot_df.shape[0]
    print(' oot_rate: ', oot_rate)
    oot_pred(model,cat_features,oot_df)


def old_tele_feature():

    allData = pd.read_csv('tele_xy_three_feature_.csv')

    print('all',allData.shape)

    feature_three_drop(allData)
	
    # 确保二分类
    allData['overdueday'] = allData['label']
	
    allData,oot_df = oot_train_split(allData)
	
    print('train',allData.shape)
    print('oot_df',oot_df.shape)
	
	# 将不参与训练的特征数据删除
    allData.drop(['order_id', 'create_time', 'overdue_days', 'label','score','obscure_score'], axis=1, inplace=True)

    # 将不参与训练的特征数据删除
    oot_df.drop(['order_id', 'create_time', 'overdue_days', 'label','score','obscure_score'], axis=1, inplace=True)
	
    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns)
                    if cont != 'overdueday']
		
    print(len(cat_features))
	
    train_rate = allData.overdueday.sum() / allData.shape[0]
    test_rate = oot_df.overdueday.sum() / oot_df.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)
	# 训练
    model = train(allData, oot_df, cat_features)


def calc_PSI(allData,oot_df,cat_features):
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

import jinpan_col
def feature_jinpan_drop(train):
    col_drop = jinpan_col.col_drop
    train.drop(col_drop, axis=1, inplace=True)

def feature_three_drop(allData):
    col_drop = ['cash_loan_15d', 'auth_contactnum_ratio_90d', 'auth_intimatenum_ratio_30d', 'org_count', 'auth_intimate_indirectnum_ratio_180d', 'datacoverge_90d', 'black_intimate_indirectnum_ratio_180d', 'match_score', 'black_indirect_peernum_ratio_90d', 'black_indirectnum_ratio_180d', 'black_intimate_indirect_peernum_ratio_30d', 'black_indirectnum_ratio_30d', 'other_count', 'auth_indirectnum_ratio_30d', 'cash_loan_30d', 'cash_loan_90d', 'consumstage_180d', 'blacklist_record_overdue_count', 'gray_record_overdue_count', 'creditScore', 'deviceRank', 'devicePrice', 'cityRec', 'provinceRec', 'countryRec', 'cityFreq', 'provinceFreq', 'countryFreq', 'ip90d', 'launchDay', 'appStability7d', 'appStability90d', 'appStability180d', 'top7d', 'top90d', 'top180d', 'tail7d', 'tail90d', 'tail180d', 'loan90d', 'loan7d', 'loan180d', 'finance7d', 'finance90d', 'finance180d', 'health7d', 'health90d', 'health180d', 'entertainment7d', 'entertainment90d', 'entertainment180d', 'tools7d', 'tools90d', 'tools180d', 'property7d', 'property90d', 'property180d', 'education7d', 'education90d', 'education180d', 'travel7d', 'travel90d', 'travel180d', 'woman7d', 'woman90d', 'woman180d', 'car7d', 'car90d', 'car180d', 'game7d', 'game90d', 'game180d', 'service7d', 'service90d', 'service180d', 'sns7d', 'sns90d', 'sns180d', 'shopping7d', 'shopping90d', 'shopping180d', 'reading7d', 'reading90d', 'reading180d', 'loan_account', 'loan_amt', 'outstand_count', 'loan_bal', 'overdue_count', 'overdue_amt', 'overdue_more_count', 'overdue_more_amt', 'generation_count', 'generation_amount', 'houmou_score', 'rnn']

    allData.drop(col_drop, axis=1, inplace=True)

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

if __name__ == '__main__':
    import time

    starttime = time.time()

    # 评估混合特征
   # mix_feature()

    #mix_all_feature()
	
    old_tele_feature()

    # extend_feature()

    # jinpan_feature()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)
