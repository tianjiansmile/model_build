
import pandas as pd
from fnmatch import fnmatch, fnmatchcase as match
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决可视化乱码问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 主要用于查看样本分布情况
# 相关性分析

# 特征分布，相关性
def data_check():
    train = pd.read_csv('data/龙井高德友盟创数据对比.csv',encoding='utf8')
    print(train.shape)
    print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    # print(pd.isnull(train).values.any())

    # 数据的整体状况
    print(train.info())

    # 检查空值比例
    # check_null = train.isnull().sum(axis=0).sort_values(ascending=False)
    # print(check_null[check_null > 0])

    # 删除任何一行有空值的记录
    # train.dropna(axis=0, how='any', inplace=True)

    # print(pd.isnull(train).values.any())

    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum']]
    print("Continuous: {} features".format(len(cont_features)))

    train['lj_score'] = train['score']
    train['ym_score'] = train['Nscore']
    train['gd_风控总分'] = train['风控总分']
    train['gd_资产系数'] = train['资产系数']
    train['gd_兴趣系数'] = train['兴趣系数']
    train['gd_高危轨迹'] = train['高危轨迹']

    # 统计离散变量的类别数
    cat_uniques = [
        'lj_score','ym_score','gd_风控总分','gd_资产系数', 'gd_兴趣系数','gd_高危轨迹'
        #         ,'最近1个月注册次数','最近3个月注册次数', '最近3个月注册次数',
        #         '最近1个月注册平台数','最近3个月注册平台数','最近12个月注册平台数',
        # 'phone_apply',
        # 'last_1_mon_total','last_3_mon_total','high_risk_total','last_12_mon_total'
        #            'comm_user_count','comm_phone_count','comm_user_phone_odd','related_user_count',
        # 'relation_age_cheat','related_age_old_exist','related_avg_degree','faith_break_users_count','same_phone_ids_count','name_fruad_hit_rate',
        # 'same_emer_unsame_names_count','same_emer_fruad_names_count','same_emer_apprear_names_count'
                   ]




    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = train[cat_uniques].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

# 特征分布，相关性
def data_check_tmp():
    train = pd.read_csv('data/lj_afu_tqy_merge.csv',encoding='utf8')
    print(train.shape)
    print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    # print(pd.isnull(train).values.any())

    # 数据的整体状况
    print(train.info())

    # 检查空值比例
    # check_null = train.isnull().sum(axis=0).sort_values(ascending=False)
    # print(check_null[check_null > 0])

    # 删除任何一行有空值的记录
    # train.dropna(axis=0, how='any', inplace=True)

    # print(pd.isnull(train).values.any())

    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum']]
    print("Continuous: {} features".format(cont_features,len(cont_features)))

    # train['lj_score'] = train['score']
    # train['ym_score'] = train['Nscore']
    # train['gd_风控总分'] = train['风控总分']
    # train['gd_资产系数'] = train['资产系数']
    # train['gd_兴趣系数'] = train['兴趣系数']
    # train['gd_高危轨迹'] = train['高危轨迹']

    # 统计离散变量的类别数
    cat_uniques = [
        'score'
       ,'阿福分','欺诈评分',
        '圈团风险浓度', '疑似准入风险', 'risk_score:'
        #         '最近1个月注册平台数','最近3个月注册平台数','最近12个月注册平台数',
        # 'phone_apply',
        # 'last_1_mon_total','last_3_mon_total','high_risk_total','last_12_mon_total'
        #            'comm_user_count','comm_phone_count','comm_user_phone_odd','related_user_count',
        # 'relation_age_cheat','related_age_old_exist','related_avg_degree','faith_break_users_count','same_phone_ids_count','name_fruad_hit_rate',
        # 'same_emer_unsame_names_count','same_emer_fruad_names_count','same_emer_apprear_names_count'
                   ]

    # print(train['阿福分(中额版)'])




    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = train[cat_uniques].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

# 某个变量的分布情况，柱状图
def missing_and_distrbute():
    train = pd.read_csv('data/钛旗云数据解析.csv', encoding='utf8')
    df = train[['圈团风险浓度']]
    # df = df.sort_values('圈团风险浓度', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    print(df.describe())
    # df.sort_values("圈团风险浓度", inplace=True)

    df['圈团风险浓度'].hist(bins=50)

    plt.show()

def common_merge():
    ym_df = pd.read_excel('data/友盟N测试结果.xlsx',encoding='utf8')
    gd_df = pd.read_excel('data/十露盘数据测试-高德测试结果.xlsx',sheetname='结果',encoding='gbk')
    lj_df = pd.read_excel('data/龙井.xlsx',encoding='utf8')

    ym_df = ym_df[['Nscore','身份证号']]
    # print(ym_df.describe())
    # print(gd_df.describe)
    gd_df = gd_df[['md5_num','风控总分','资产系数','兴趣系数','高危轨迹']]

    print('gd_df', gd_df.shape)
    print('ym_df', ym_df.shape)
    print('lj_df', lj_df.shape)

    ym_df['md5_num'] = ym_df['身份证号']
    lj_df['md5_num'] = lj_df['身份证号']

    # 横向合并
    all_df = pd.merge(lj_df, ym_df, on='md5_num', how='inner')
    all_df = pd.merge(all_df, gd_df, on='md5_num',how='inner')

    print('all_df', all_df.shape)

    all_df.to_csv('data/龙井高德友盟创数据对比.csv', index=False)

def common_merge1():
    # ym_df = pd.read_excel('data/友盟N测试结果.xlsx',encoding='utf8')
    # gd_df = pd.read_excel('data/十露盘数据测试-高德测试结果.xlsx',sheetname='结果',encoding='gbk')
    lj_df = pd.read_excel('data/龙井.xlsx',encoding='utf8')
    int_df = pd.read_csv('data/int_20w_new.csv', encoding='utf8')

    # ym_df = ym_df[['Nscore','身份证号']]
    # # print(ym_df.describe())
    # # print(gd_df.describe)
    # gd_df = gd_df[['md5_num','风控总分','资产系数','兴趣系数','高危轨迹']]

    print('int_df', int_df.shape)
    print('lj_df', lj_df.shape)

    int_df['md5_num'] = int_df['md5_id_num']
    lj_df['md5_num'] = lj_df['身份证号']

    lj_df = lj_df[['md5_num','score']]

    # 横向合并
    all_df = pd.merge(int_df, lj_df, on='md5_num', how='inner')
    # all_df = pd.merge(all_df, gd_df, on='md5_num',how='inner')

    print('all_df', all_df.shape)
    all_df['label'] = all_df['pd7']

    all_df.to_csv('data/龙井_label.csv', index=False)

def zdxm_merge():
    # bqs_df = pd.read_csv('D:/三方数据测试/白骑士/bqs_feature.csv', encoding='utf8')
    ali_df = pd.read_excel('D:/三方数据测试/third_test0912/阿里1125/ali_time_back_201904-201907_test.xlsx', encoding='utf8',sheetname='data')
    afu_df = pd.read_excel('D:/三方数据测试/third_test0912/致诚阿福/zcaf_data.xlsx', encoding='utf8')

    # bqs_col = [n for n in bqs_df.columns
    #            if n not in ['close_time','order_id','overdue_days','repay_date','update_time','label']]
    # for col in bqs_col:
    #     bqs_df['BQS_'+col] = bqs_df[col]
    # bqs_df.drop(bqs_col, axis=1, inplace=True)


    ali_df['ALi_risk_type'] = ali_df['risk_type']
    ali_df['ALi_score_nt'] = ali_df['score_nt']
    ali_df = ali_df[['order_id','ALi_score_nt','ALi_risk_type']]

    # afu_df.drop(['身份证','手机号','create_time','md5_mobile','name','product_name','overdue_days','label','loan'], axis=1, inplace=True)
    pub_col = ['身份证','手机号','create_time','md5_mobile','name','product_name','overdue_days','label','loan','order_id']
    # af_col = [n for n in afu_df.columns if n not in  pub_col]
    # for col in af_col:
    #     afu_df['AFU_' + col] = afu_df[col]
    # afu_df.drop(af_col, axis=1, inplace=True)

    all_df = pd.merge(afu_df, ali_df, on='order_id', how='left')
    print(afu_df.shape, '加阿里',all_df.shape)
    # all_df = pd.merge(all_df, afu_df, on='order_id', how='left')
    # print('加阿福',all_df.shape)



    all_df.to_csv('D:/智度小贷/ali_afu_feature.csv', index=False)

def common_merge2():
    # ym_df = pd.read_excel('data/友盟N测试结果.xlsx',encoding='utf8')
    # gd_df = pd.read_excel('data/十露盘数据测试-高德测试结果.xlsx',sheetname='结果',encoding='gbk')
    tqy_id_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx',encoding='utf8',sheetname='申请id')
    tqy_afrud_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='反欺诈')
    tqy_phone_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='申请phone')
    # tqy_gz_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='共债')
    # tqy_yq_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='逾期')
    tqy_qb_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='圈团')
    tqy_ky_df = pd.read_excel('D:/三方数据测试/钛旗云/钛旗云测试/多头/反欺诈&多头20191230.xlsx', encoding='utf8', sheetname='可疑')
    int_df = pd.read_csv('data/int_20w_new.csv', encoding='utf8')

    # ym_df = ym_df[['Nscore','身份证号']]
    # print(ym_df.describe())
    # print(gd_df.describe)
    # gd_df = gd_df[['md5_num','风控总分','资产系数','兴趣系数','高危轨迹']]

    tqy_afrud_df = tqy_afrud_df[['risk_score:','id_card_encrypted']]
    tqy_id_df.drop(['app_date','name'], axis=1, inplace=True)
    tqy_phone_df.drop(['app_date', 'phonenum_encrypted', 'name'], axis=1, inplace=True)
    # tqy_gz_df.drop(['app_date', 'phonenum_encrypted', 'name'], axis=1, inplace=True)
    # tqy_yq_df.drop(['app_date', 'phonenum_encrypted', 'name'], axis=1, inplace=True)
    tqy_qb_df.drop(['app_date', 'phonenum_encrypted', 'name'], axis=1, inplace=True)
    tqy_ky_df.drop(['app_date', 'phonenum_encrypted', 'name'], axis=1, inplace=True)

    print('int_df', int_df.shape)
    print('tqy_id_df', tqy_id_df.shape)

    int_df['id_card_encrypted'] = int_df['md5_id_num']
    # tqy_df['md5_num'] = tqy_df['id_card_encrypted']

    int_df['label'] = int_df['pd7']
    int_df = int_df[['id_card_encrypted','label','product_name','create_time']]

    # 横向合并
    all_df = pd.merge(tqy_id_df, tqy_phone_df, on='id_card_encrypted', how='inner')
    all_df = pd.merge(all_df, tqy_afrud_df, on='id_card_encrypted', how='inner')
    all_df = pd.merge(all_df, tqy_qb_df, on='id_card_encrypted', how='inner')
    all_df = pd.merge(all_df, tqy_ky_df, on='id_card_encrypted', how='inner')
    all_df = pd.merge(all_df, int_df, on='id_card_encrypted', how='inner')
    # all_df = pd.merge(all_df, gd_df, on='md5_num',how='inner')

    print('all_df', all_df.shape)
    # all_df['label'] = all_df['pd7']

    all_df.to_csv('D:/三方数据测试/钛旗云/钛旗云测试/多头/tqy_2w_label.csv', index=False)

def tmp_merge():
    # lj_df = pd.read_excel('D:/三方数据测试/龙井分/lj_20191231（分数测试结果）.xlsx', encoding='utf8')
    lj_df = pd.read_csv('data/龙井_label.csv', encoding='utf8')

    ydun_df = pd.read_excel('D:/三方数据测试/third_test0912/第四季度/千云测试/十露盘-先享后付-测试结果-20191118.xlsx', encoding='utf8')

    lj_df = lj_df[['md5_num','score','create_time','product_name','label']]
    ydun_df = ydun_df[['cert_no_md5','yidun_score_then']]

    lj_df['cert_no_md5'] = lj_df['md5_num']

    all_df = pd.merge(lj_df, ydun_df, on='cert_no_md5', how='inner')

    cat_uniques = [
        'yidun_score_then', 'score'
        # ,'欺诈等级','评分等级'
        #         '最近1个月注册平台数','最近3个月注册平台数','最近12个月注册平台数',
        # 'phone_apply',
        # 'last_1_mon_total','last_3_mon_total','high_risk_total','last_12_mon_total'
        #            'comm_user_count','comm_phone_count','comm_user_phone_odd','related_user_count',
        # 'relation_age_cheat','related_age_old_exist','related_avg_degree','faith_break_users_count','same_phone_ids_count','name_fruad_hit_rate',
        # 'same_emer_unsame_names_count','same_emer_fruad_names_count','same_emer_apprear_names_count'
    ]

    print(all_df.shape)

    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = all_df[cat_uniques].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

    all_df.to_csv('data/lj_yidun_merge.csv', index=False)

def tmp_merge1():
    lj_df = pd.read_excel('D:/三方数据测试/龙井分/龙井分测试结果1.5.xlsx', encoding='utf8')
    lj_df_label = pd.read_excel('D:/三方数据测试/龙井分/sample_1.5_label.xlsx', encoding='utf8')
    tqx_df = pd.read_csv('D:/三方数据测试/钛旗云/钛旗云测试/多头/tqy_2w_label.csv',encoding='utf8')

    tqx_df['md5_id_num'] = tqx_df['id_card_encrypted']
    tqx_df = tqx_df[['md5_id_num','互金_身份证1周内申请次数',
                     '互金_身份证1月内申请平台数','互金_身份证3月内申请次数',
                     '互金_身份证3月内申请平台数','互金_身份证6月内申请次数',
                     '互金_身份证6月内申请平台数','互金_身份证12月内申请次数',
                     '互金_身份证12月内申请平台数','圈团风险浓度','疑似准入风险','risk_score:']]

    # ydun_df = pd.read_excel('D:/三方数据测试/third_test0912/第四季度/千云测试/十露盘-先享后付-测试结果-20191118.xlsx', encoding='utf8')

    # lj_df['md5_id_num'] = lj_df['身份证号']
    lj_df = lj_df[['md5_id_num','score']]

    all_df = pd.merge(lj_df, lj_df_label, on='md5_id_num', how='inner')

    # all_df.to_csv('data/lj_new_1.5.csv', index=False)

    afu_df = pd.read_excel('D:/三方数据测试/third_test0912/致诚阿福/zcaf_data.xlsx', encoding='utf8')

    afu_df['md5_id_num'] = afu_df['身份证']
    afu_df = afu_df[['md5_id_num','欺诈评分','阿福分']]

    # ali_df = pd.read_excel('D:/三方数据测试/third_test0912/阿里1125/ali_time_back_201904-201907_test.xlsx', encoding='utf8',
    #                        sheetname='data')
    # ali_df['ALi_risk_type'] = ali_df['risk_type']
    # ali_df['ALi_score_nt'] = ali_df['score_nt']
    # ali_df = ali_df[['order_id', 'ALi_score_nt', 'ALi_risk_type']]

    all_df = pd.merge(afu_df, all_df, on='md5_id_num', how='left')
    all_df = pd.merge(all_df, tqx_df, on='md5_id_num', how='left')

    print('lj_df',lj_df.shape,'afu_df',afu_df.shape,'tqx_df',tqx_df.shape,'merge shape', all_df.shape)
    all_df.to_csv('data/lj_afu_tqy_merge.csv', index=False)


if __name__ == '__main__':
    # common_merge2()

    # 相关性情况
    # data_check()

    # zdxm_merge()

    tmp_merge1()

    # data_check_tmp()
