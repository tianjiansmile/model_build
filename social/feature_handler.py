import pandas as pd
from fnmatch import fnmatch, fnmatchcase as match
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 特征数据合并


def extend_clean(extend_df):
    print(extend_df.shape)
    num_features = [cont for cont in list(extend_df.select_dtypes(
        include=['float64', 'int64']).columns) if
                    'region' not in cont and
                    'contact_list_cnt_phoneNumLoc' not in cont and
                    'contact_list_ratioOfCnt_phoneNumLoc' not in cont
                    ]
    # print(num_features)
    # print(len(num_features))

    # 非数字类特征盘点
    txt_features = [cont for cont in list(extend_df.select_dtypes(
        exclude=['float64', 'int64']).columns) if
                    cont not in ['EI_score_友盟分','EI_nifadata_loanamt']]

    # print(txt_features)
    print('txt len',len(txt_features))

    # 离散文本类特征
    cate_text_fea = [cont for cont in list(extend_df.select_dtypes(
        exclude=['float64', 'int64']).columns) if
                     # 'tanzhidata_mb_infos_lastweek' in cont or
                     # 'tanzhidata_refInfos' in cont and
                     'level' in cont and
                     cont not in ['EI_score_友盟分', 'EI_nifadata_loanamt']]

    # print(cate_text_fea)

    # num_features.extend(cate_text_fea)

    # 查看该文本特征的分类
    # print(extend_df['EI_tanzhidata_mb_infos_lastweek5_overdue_average_level'].unique())
    # # 转换
    # extend_df['EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level'] = pd.factorize(
    #     extend_df["EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level"])[0].astype(np.uint16)
    #
    # print(extend_df['EI_tanzhidata_mb_infos_lastweek0_apply_request_average_level'].unique())

    return extend_df[num_features]




def common_merge():
    shuc_df = pd.read_excel('third_part_data/slp回溯样本16w多头测试结果20191224.xlsx',encoding='utf8')
    xb_df = pd.read_excel('third_part_data/小白信用_短贷评分_十露盘_20191202161002.xlsx',encoding='utf8')

    xb_df = xb_df[['手机号','最近1个月验证码通知平台数','最近3个月验证码通知平台数','最近12个月验证码通知平台数'
                ,'最近1个月验证码通知次数', '最近3个月验证码通知次数','最近12个月验证码通知次数'
                ,'最近1个月注册次数','最近3个月注册次数', '最近3个月注册次数',
                '最近1个月注册平台数','最近3个月注册平台数','最近12个月注册平台数']]
    print('shuc_df', shuc_df.shape)
    print('xb_df', xb_df.shape)

    xb_df['md5_mobile'] = xb_df['手机号']

    # 横向合并
    all_df = pd.merge(shuc_df, xb_df, on='md5_mobile',how='inner')

    print('all_df', all_df.shape)

    all_df.to_csv('third_part_data/京东小白数创数据对比.csv', index=False)



def label_map(x):
    if x >= 1:
        return 1
    else:
        return 0

from datetime import datetime
import time
def oot_train_split(allData):
    create_time = allData['create_time']
    create_time = create_time.sort_values()
    create_time.index = range(len(create_time))
    oot_index = round(len(create_time) * 0.9)
    oot_time = create_time[oot_index]

    allData = allData.assign(data_set=allData['create_time'].apply(lambda x: 'train' \
        if datetime.strptime(x[0:9], "%Y-%m-%d") <= datetime.strptime(oot_time[0:9], "%Y-%m-%d") \
        else 'test'))

    oot_df = allData[allData['data_set'] == 'test']
    train_df = allData[allData['data_set'] == 'train']

    return train_df, oot_df

def oot_train_split_pro(allData):

    allData['split'] = allData['create_time'].apply(lambda x: 'oot' if x >= '2019-5-21' else 'train')
    groupby = allData.groupby(allData['split'])
    for name, group in allData.groupby('split'):
        if name is 'oot':
            oot = group
        else:
            train = group

    return train, oot


def data_check():
    predData = pd.read_csv('social/data/social_filter_user_2019_test2.csv'
                           ,error_bad_lines=False,encoding='gbk')
    # col_ = [cont for cont in list(predData.columns) if
    #         'EI_' in cont
    #         ]
    #
    # predData = predData[col_]

    print(predData.shape)



    train_df, oot_df = oot_train_split_pro(predData)

    print(train_df.shape,oot_df.shape)

def data_map():
    train = pd.read_csv('C:/Users/EDZ/Desktop/social_emer_2019_test3.csv')

    train['m1'] = train['m1_times'].map(label_map)

    train.to_csv('social/data/social_emer_2019_test3_pro.csv')

def emer_data_plot():
    train = pd.read_excel('C:/Users/EDZ/Desktop/emer_fea_fruad_table.xlsx')
    # 2 times
    # train.plot(y=['avg_2_times',
    #               'phone_count_gt_3',
    #               'name_fruad_hit_count_gt_4'
    #               ,'same_emer_cheat_names_count_gt_2',
    #               'related_emer_phone_count_gt_3',
    #               'is_ph_em_same_gt_2',
    #               ])

    # 2 times 交叉
    # train.plot(y=['avg_2_times',
    #               'names_count_gt_8',
    #               'emer_phone_count_gt_8'
    #     , 'is_ph_em_same_eq_1',
    #               'name_fruad_hit_count_gt_4',
    #               'rel_unsame_times_gt_13'])

    # 2 times 交叉
    train.plot(y=['avg_2_times',
                  'avg_1.8_times',
                  'comm_phone_count_bt_20_300',
                  'comm_phone_count_bt_10_19'
])
    plt.show()

def emer_graph_vec_merge():
    e3_df = pd.read_csv('data/social_emer_2019_graph_vec.csv')

    e3_df.drop(['create_time','order_id','m1_enev','first_overdue_days',
                 'product_id','slpd7','m1_times','m2','m3','m4','m5'
                ], axis=1, inplace=True)

    e4_df = pd.read_csv('data/social_emer_2019_test6.csv')

    print('e3_df', e3_df.shape)
    print('e4_df', e4_df.shape)

    # 横向合并
    all_df = pd.merge(e4_df,e3_df, on='md5_num',how='inner')

    print('mix all ', all_df.shape)

    all_df.to_csv('social_emer_2019_graph_merge.csv', index=False)


# 相关性分析
def data_check():
    train = pd.read_csv('data/钛旗云数据解析.csv',encoding='utf8')
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
                    # '圈团风险浓度',
                   # '疑似准入风险',
                   # '疑似金融黑产',
                   # '疑似网络违规','疑问羊毛党',
        '手机号疑似为多个用户公用(51003)',
                   'phone_count','emer_phone_count','names_count','relation_count','is_ph_em_same','name_len_two','rel_repel_count',
        #            'comm_user_count','comm_phone_count','comm_user_phone_odd','related_user_count',
        # 'relation_age_cheat','related_age_old_exist','related_avg_degree','faith_break_users_count','same_phone_ids_count','name_fruad_hit_rate',
        # 'same_emer_unsame_names_count','same_emer_fruad_names_count','same_emer_apprear_names_count'
                   ]


    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = train[cat_uniques].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

def missing_and_distrbute():
    train = pd.read_csv('data/钛旗云数据解析.csv', encoding='utf8')
    df = train[['圈团风险浓度']]
    # df = df.sort_values('圈团风险浓度', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    print(df.describe())
    # df.sort_values("圈团风险浓度", inplace=True)

    df['圈团风险浓度'].hist(bins=50)

    plt.show()

def plot_check(df ):
    df = df.set_index('date')
    # df['int_count'].plot()
    # df['pdl_count'].plot()
    # df['sum_count'].plot()
    # plt.show()
    df.plot()
    plt.show()

if __name__ == '__main__':
    import time
    starttime = time.time()

    # tele_ex_jin_addr_merge()
    common_merge()

    # data_check()

    # xinyan_df = pd.read_csv('jk_0608_xy_feautre_int.txt')
    # xinyan_new_df = pd.read_csv('online_xy_feature_recall.txt')
    #
    # xinyan_df = xinyan_clean(xinyan_df, xinyan_new_df)

    # emer_merge()

    # data_map()

    # emer_graph_vec_merge()

    # missing_and_distrbute()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)

