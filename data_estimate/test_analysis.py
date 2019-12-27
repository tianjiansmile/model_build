import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import sys

import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import model_selection
from scipy.stats import ks_2samp
import seaborn as sns

##############################################################################
#  主要用于数据评估
#  1 IV值
#  2 KS值
#  3 AUC
##############################################################################


from data_estimate.handle_utils import analysis
exam = analysis()
#######################################数据预处理#######################################
df_org=pd.read_csv(r'merge_data_final.csv',encoding='gbk',low_memory=False)
print(df_org.shape)

# #数据类型查看
# type=df_org.dtypes
# # df_org['create_time']=df_org['create_time'].astype('object')
# type.to_csv(r'D:\yinni\type.csv')
#缺失值查看
missing_df = df_org.isnull().sum(axis =0).reset_index()
fea_miss = pd.DataFrame(1-df_org.count()/df_org.shape[0]).reset_index()
fea_miss.to_csv(r'missing.csv')
#填充缺失值
df_org=df_org.fillna(value=-1)
fea_col = list(df_org.columns)
dele_list = ['app_date','md5_mobile','最频繁乘机出发城市','最频繁乘机到达城市','最频繁使用航空公司以及乘机次数',\
             '最后起飞城市','最后抵达城市','product_name','md5_num','label','create_time','loan']
for fea in dele_list:
    try:
        fea_col.remove(fea)
    except:
        continue
print(fea_col)

#测试变量数据集
use_col=[]
use_col.extend(fea_col)
use_col.extend(['label','product_name'])
df_test=df_org[use_col]
df_test.to_csv(r'df_test.csv')
#产品列表
pro_lst=list(set(df_org['product_name'].tolist()))

######################################单变量分析#######################################
 # 计算各变量IV值(不分产品)
fea_iv = pd.DataFrame([[0 for i in range(2)] for j in range(len(fea_col))], columns=['feature', 'iv'])
for i in range(len(fea_col)):
    fea = fea_col[i]
    #fea='身份证关联到的7天内租赁行业下申贷事件中的记录数'
    fea_iv.iloc[i, 0] = fea
    df_iv = df_org[[fea, 'label']]
    woe_iv = pd.DataFrame({fea: df_iv[fea], "Bucket": exam.Equal_Freqency(df_iv[fea], 10), 'label': df_iv['label']})
    df_out = exam.woe_iv_calculate(woe_iv, fea, 'label', bucket='Bucket')
    iv = df_out['IV'].sum()
    fea_iv.iloc[i, 1] = iv

#iv值导出
fea_iv=fea_iv.sort_values(by="iv", ascending=False)
fea_iv.to_csv('iv.csv',index=None)

# 计算各变量IV值(分产品)
fea_iv_pro = []
for pro in pro_lst:
    df_pro = df_org.loc[df_org['product_name']==pro]
    for i in range(len(fea_col)):
        try:
            fea = fea_col[i]
            df_iv = df_pro[[fea, 'label']]
            woe_iv = pd.DataFrame({fea: df_iv[fea], "Bucket": exam.Equal_Freqency(df_iv[fea], 10), 'label': df_iv['label']})
            df_out = exam.woe_iv_calculate(woe_iv, fea, 'label', bucket='Bucket')
            iv = df_out['IV'].sum()
            temp_lst=[pro,fea,iv]
            fea_iv_pro.append(temp_lst)
        except:
            continue
#列表转数据框
fea_iv_df = pd.DataFrame(fea_iv_pro, columns=['pro','feature','iv'])
#iv值导出
fea_iv_df=fea_iv_df.sort_values(by=["iv",'pro'],ascending=False)
fea_iv_df.to_csv(r'iv_pro.csv')

# # #iv异常变量查看
# # fea_temp='CPL0045'
# # df_org[[fea_temp,'product_name']].to_csv(r'fea.csv',encoding="gbk")
# # #单变量分析
# # df_org[['CPL0013','label','loan','product_name']].to_csv(r'dan.csv',encoding="gbk")
#
#######################################ks计算#######################################
def get_ks(df, bin_num=10):
    if df.shape[0] == 0:
        return [], [], -1, -1, []

    df = copy.deepcopy(df)
    f_uniq = copy.deepcopy(df['f'].drop_duplicates().get_values())
    f_uniq.sort()

    if len(f_uniq) <= bin_num:
        df['group'] = df['f']
        bin_value = list(f_uniq)
        bin_value.sort()
    else:
        f_series = sorted(df['f'].get_values())
        f_cnt = len(f_series)
        bin_ratio = np.linspace(1.0 / bin_num, 1, bin_num)
        bin_value = list(set([f_series[int(ratio * f_cnt) - 1] for ratio in bin_ratio]))
        bin_value.sort()
        if f_series[0] < bin_value[0]:
            bin_value.insert(0, f_series[0])
        df['group_'] = pd.cut(df['f'], 10, retbins=False, include_lowest=True, duplicates='drop')
        df['group'] = [str(group) for group in df['group_']]
    del df['f']
    group_info = df.groupby('group')
    group_sum_info = group_info.sum()
    if group_sum_info.shape[0] == 0:
        return [], [], -1, -1, []
    cumsum_info = group_sum_info.cumsum()

    group_sum_info['total'] = group_sum_info['good'] + group_sum_info['bad']
    total_good = sum(group_sum_info['good'])
    total_bad = sum(group_sum_info['bad'])
    total = total_good + total_bad
    group_sum_info['sample_ratio'] = round(group_sum_info['total'] / total,7)
    group_sum_info['bad_ratio'] = round(group_sum_info['bad'] / total_bad,7)
    group_sum_info['good_ratio'] = round(group_sum_info['good'] / total_good,7)

    group_sum_info['cum_good'] = round(cumsum_info['good'],7)
    group_sum_info['cum_bad'] = round(cumsum_info['bad'],7)
    group_sum_info['cum_good_ratio'] = round(group_sum_info['cum_good'] / total_good,7)
    group_sum_info['cum_bad_ratio'] = round(group_sum_info['cum_bad'] / total_bad,7)
    tmp = group_sum_info['cum_good_ratio'] - group_sum_info['cum_bad_ratio']
    group_sum_info['ks'] = ['%.6f' % abs(value) for value in tmp]

    ks = max(group_sum_info['ks'])

    result = []
    result.append(
        u'range\ttotal_cnt\tbad_cnt\tbad_pc%\tgood_cnt\tgood_pc%\tcum_bad_ratio\tcum_good_ratio\tKS_')
    for row in range(len(group_sum_info)):
        out_list = [group_sum_info.index[row]]
        for key in ['total', 'bad', 'bad_ratio', 'good', 'good_ratio', 'cum_bad_ratio',
                    'cum_good_ratio', 'ks']:
            out_list.append(group_sum_info[key].iloc[row])
        result.append('\t'.join([str(item) for item in out_list]))
    return group_sum_info, result, ks, bin_value

def get_feature_ks(df_in, feature_name, label_name):

    df = df_in[[feature_name, label_name]].reset_index(drop=True)
    # 计算ks,    pd_df是x+label列,x是单特征
    tmp_df = pd.DataFrame()
    tmp_df['good'] = 1 - df[label_name]
    tmp_df['bad'] = df[label_name]
    tmp_df['f'] = df[feature_name]
    group_info, result, ks, bin_value = get_ks(df=tmp_df, bin_num=10)

    array = []
    for line in result:
        line = line.split('\n')[0]
        line = line.split('\t')
        array.append(line)

    columns = [u'range', u'total_cnt', u'bad_cnt', u'bad_pc%', u'good_cnt', u'good_pc%', u'cum_bad_ratio',
               u'cum_good_ratio', u'KS_']

    KS_table = pd.DataFrame(array[1:], columns=columns).sort_values(by='KS_', ascending=False)
    return KS_table

def ks(df_ks, pro_ks,features=[], if_fillna=1, fill_value=-1, label_name='label'):
    """
    ks计算
    :param df: dataframe
    :param features: 待计算特征
    :param if_fillna: 是否填充缺失值，1为填充，填充值为fill_value的值，默认为-1,0为不填充
    :param fill_value: 填充值
    :return:
    """

    if if_fillna != 1 and if_fillna != 0:
        raise Exception("if_fillna can olny be 1 or 0")

    valid_features = features
    # print('len(valid_features):', len(valid_features))
    result_table = []

    if if_fillna == 1:
        df_tmp = df_ks.fillna(fill_value).reset_index(drop=True)
        for col in valid_features:
            try:
                df_tmp[col] = df_tmp[col].astype('float')
                KS_table = get_feature_ks(df_tmp, col, label_name)
                KS_table = KS_table.sort_index()
                print(pro_ks, col, ' KS:', max(KS_table['KS_']))
                print(KS_table)
                result_table.append([pro_ks, col, max(KS_table['KS_'])])
            except:
                info = sys.exc_info()

    else:
        for col in valid_features:
            try:
                df_tmp = df_ks.dropna(subset=[col], how='any', axis=0).reset_index(drop=True)
                df_tmp[col] = df_tmp[col].astype('float')
                KS_table = get_feature_ks(df_tmp, col, label_name)
                #print(KS_table)
                KS_table = KS_table.sort_index()
                print(pro_ks,col, ' KS:', max(KS_table['KS_']))
                print(KS_table)
                result_table.append([pro_ks,col, max(KS_table['KS_'])])

            except:
                continue

    result_table = pd.DataFrame(data=result_table, columns=['pro','feature', 'KS'])
    result_table['KS'] = result_table['KS'].astype('float')
    result_table = result_table.sort_values(by='KS', ascending=False).reset_index(drop=True)

    return result_table

#临时存储ks结果
ks_lst=[]
for pro_ks in pro_lst:
    df_ks = df_org.loc[df_org['product_name']==pro_ks]
    #result_table = ks(df,features=df.iloc[:,0:-1],if_fillna=0,fill_value=-1,label_name='label')
    result_table = ks(df_ks,pro_ks,features=fea_col,if_fillna=0,fill_value=-1,label_name='label')
    ks_lst.append(result_table)

ks_result=pd.concat(ks_lst)
ks_result.to_csv(r'ks.csv')
#######################################跑模型#######################################

def cal_bad_rate(df, date_col_name='create_time', label_name='label', by='month'):
    """
    按天切片计算违约率
    :param df:原始数据
    :param date_col_name: 日期列名,精确到天 eg:2019-04-15 ,也可以精确到月,eg:2019-04
    :param label_name:  标签的列名
    :param by:  按月/按天
    :return:   df 日期-总数-坏账率
    """
    if by == 'month':
        df['date'] = df[date_col_name].apply(lambda x: str(x)[:6])
    else:
        df['date'] = df[date_col_name].apply(lambda x: str(x)[:8])
    r_df = pd.DataFrame(columns=['date', 'total_amt','bad_rate'])
    r_ls = list()
    for date in np.unique(df['date']):
        total_amt = df[df['date'] == date].shape[0]
        bad_amt = df[df['date'] == date][label_name].sum()
        r_ls = [date, total_amt, round(float(bad_amt/total_amt), 3)]
        r_df.loc[len(r_df)] = r_ls
    return r_df
bad_rate = cal_bad_rate(df=df_org)
print(bad_rate)

def splitbyTime(df, date_col_name, date_time):
    print('origin data size:%d'%(df.shape[0]))
    print('after %s data size:%d'%(date_time, sum(df[date_col_name] >= date_time)))
    df1 = df[df[date_col_name] < date_time]
    df2 = df[df[date_col_name] >= date_time]
    return(df1, df2)

def getAUC(model, dfTrain_X, y_train, dfValid_X, y_valid,dfTest_X, y_test):
    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_auc = roc_auc_score(y_train.astype("int"), data_predict_1train)
    data_predict_1valid = model.predict(dfValid_X.astype("float"))
    valid_auc = roc_auc_score(y_valid.astype("int"), data_predict_1valid)
    data_predict_1test = model.predict(np.array(dfTest_X.astype("float")))
    test_auc = roc_auc_score(y_test.astype("int"), data_predict_1test)
    if test_auc<0.5:
        test_auc=1-test_auc
    print("train_auc:%f \nval_auc:%f \noot_auc:%f"%(train_auc, valid_auc,test_auc))

def compute_ks(model, dfTrain_X, y_train, dfValid_X, y_valid,dfTest_X, y_test):
    get_ks = lambda proba, label:ks_2samp(proba[label == 1], proba[label != 1]).statistic
    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_ks = get_ks(data_predict_1train, y_train.astype("int"))
    data_predict_1valid = model.predict(dfValid_X.astype("float"))
    valid_ks = get_ks(data_predict_1valid, y_valid.astype("int"))
    data_predict_1test = model.predict(np.array(dfTest_X.astype("float")))
    test_ks = get_ks(data_predict_1test, y_test.astype("int"))

    print("train_KS:%f\nval_KS:%f\noot_ks:%f"% (train_ks, valid_ks,test_ks))

def feature_importance(pro,model, column_name):

    feat_imp = pd.DataFrame(
        dict(columns=column_name, feature_importances=model.feature_importance() / model.feature_importance().sum()))
        #dict(columns=column_name, feature_importances=model.feature_importance()))
    feat_imp2 = feat_imp.sort_values(by=["feature_importances"], axis=0, ascending=False).head(100)
    feat_imp2['pro']=pro
    feat_imp2=feat_imp2.rename(columns={'columns': 'feature'})
    return (feat_imp2)

def lgb_model(X_train, X_valid, y_train, y_valid, params):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    models = lgb.train(params=params,
                      train_set=lgb_train,
                      num_boost_round=50000,
                      valid_sets=[lgb_train, lgb_eval],
                      early_stopping_rounds=4000,
                      verbose_eval=100)
    # models.save_model('/home/wangyuanjiang/thirdpartydataTest/suanhua/model/'+model_name+'.model')
    return (models)

#参数设置
params = {
    'task': 'train',
    'boosting_type': 'gbdt', # GBDT算法为基础
    'objective': 'binary', # 因为二分类，所以是binary，0，1
    'metric': 'auc',# 评判指标
    'num_leaves': 63,# 大会更准,但可能过拟合
    'learning_rate': 0.001,# 学习率
    'feature_fraction': 0.4,# 防止过拟合
    'bagging_fraction': 0.5,# 防止过拟合
    'bagging_freq': 10,# 防止过拟合
    'verbose': -1,
    'num_threads': 6,
    'min_data_in_leaf': 200,# 防止过拟合
    'lambda_l1': 0.1}
#分产品跑模型
#临时存储特征重要性结果
fea_imp=[]
for pro in pro_lst:
    print('**************************************')
    print('%s 开始训练：' %pro)
    df_pro_model = df_org.loc[df_org['product_name']==pro]
    #坏账率
    bad_rate = cal_bad_rate(df=df_pro_model)
    #测试集大约占10%
    len_model=df_pro_model.shape[0]
    temp_loc=int(len_model/10)*9
    #获取切片时间
    cut_time=df_pro_model.sort_values(by='create_time',ascending=True).iloc[temp_loc,:]['create_time']
    print(cut_time)
    # cut_time = pd.qcut(df_pro_model['create_time'], 10,retbins=True)
    df_model, df_time = splitbyTime(df_pro_model, 'create_time', cut_time)
    #训练集与验证集
    X_train, X_valid, y_train, y_valid = \
    model_selection.train_test_split(np.array(df_pro_model[fea_col]),df_pro_model['label'].values,test_size=0.10,random_state=26,stratify=df_pro_model.label)
    print('训练集大小：{}，坏账率：{} '.format(X_train.shape[0], float(sum(y_train)/X_train.shape[0])))
    print('验证集大小：{}，坏账率：{} '.format(X_valid.shape[0], float(sum(y_valid)/X_valid.shape[0])))
    print('测试集大小：{}，坏账率：{} '.format(df_time.shape[0], float(sum(df_time.label)/df_time.shape[0])))
    #开始训练模型
    dhb_model = lgb_model(X_train, X_valid, y_train, y_valid, params)
    #AUC比较
    getAUC(model=dhb_model, dfTrain_X=X_train, y_train=y_train,
           dfValid_X=X_valid, y_valid=y_valid,dfTest_X=df_time[fea_col], y_test=df_time['label'])

    #KS比较
    compute_ks(model=dhb_model, dfTrain_X=X_train, y_train=y_train,
           dfValid_X=X_valid, y_valid=y_valid,dfTest_X=df_time[fea_col], y_test=df_time['label'])
    #输出特征重要性
    feature_imp=feature_importance(pro,dhb_model, fea_col)
    fea_imp.append(feature_imp)
    print(fea_imp)

fea_imp_result = pd.concat(fea_imp)
fea_imp_result.to_csv(r'fea_imp_result.csv')
#######################################合并变量结果#######################################
fea_result=fea_iv_df.merge(ks_result,how='inner',on=['pro','feature'])
fea_result_all=fea_result.merge(fea_imp_result,how='outer',on=['pro','feature'])
fea_result_all.to_csv(r'fea_result_all.csv')

#######################################规则之间重合度#######################################
#df_org = pd.read_excel(r'data.xlsx',encoding="gbk",sheet_name='Sheet1')
#df_org1 =df_org.loc[df_org['bad']!=1]
#pro_lst=list(set(df_org['product_name'].tolist()))

#rule_lst=["df_org2['欺诈评分']>72","df_org2['阿福分(中额版)']<400","df_org2['评分等级']=='E'",
#          "df_org2['欺诈等级'].isin(['四级','五级'])","df_org2['长期拖欠']==1","df_org2['小额业务获批困难']==1",
#          "df_org2['模型评估低资质']==1"]

#for rule in rule_lst:
#    print(rule)
#
#from itertools import combinations
#compare_lst=list(combinations(rule_lst, 2))
#
#for cmp in compare_lst:
#    print(cmp[0])
#    print(cmp[1])
#
#
#for pro in pro_lst:
#    print(pro)
#    for cmp in compare_lst:
#        print('%s与%s做对比' % (cmp[0],cmp[1]) )
#        df_org2=df_org1.loc[(df_org1['product_name']=='秒啦')&(df_org1['label']==1)]
        #规则筛选
#        df1=df_org2.loc[eval(cmp[0])]
#        df2=df_org2.loc[eval(cmp[1])]
#        df3=df1.merge(df2,how='inner',on='order_id')
#        mm=len(df3)
#        nn=min(len(df1),len(df2))
#        print(mm/nn)
