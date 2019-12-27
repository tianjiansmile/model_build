
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
    train = pd.read_csv('data/京东小白数创数据对比.csv',encoding='utf8')
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

    # 统计离散变量的类别数
    cat_uniques = [
        '最近1个月验证码通知平台数','最近3个月验证码通知平台数','最近12个月验证码通知平台数'
                ,'最近1个月验证码通知次数', '最近3个月验证码通知次数','最近12个月验证码通知次数'
                ,'最近1个月注册次数','最近3个月注册次数', '最近3个月注册次数',
                '最近1个月注册平台数','最近3个月注册平台数','最近12个月注册平台数',
        'phone_apply',
        'last_1_mon_total','last_3_mon_total','high_risk_total','last_12_mon_total'
        #            'comm_user_count','comm_phone_count','comm_user_phone_odd','related_user_count',
        # 'relation_age_cheat','related_age_old_exist','related_avg_degree','faith_break_users_count','same_phone_ids_count','name_fruad_hit_rate',
        # 'same_emer_unsame_names_count','same_emer_fruad_names_count','same_emer_apprear_names_count'
                   ]


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


if __name__ == '__main__':
    # 相关性情况
    data_check()
