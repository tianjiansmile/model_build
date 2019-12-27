import pandas as pd
from fnmatch import fnmatch, fnmatchcase as match
import numpy as np
import matplotlib.pyplot as plt


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

def jinpan_concat():
    jinpan_2 = pd.read_csv('data/jiaka_jinpan_feature_range(0, 20000).txt')
    jinpan_4 = pd.read_csv('data/jiaka_jinpan_feature_range(20000, 40000).txt')
    jinpan_6 = pd.read_csv('data/jiaka_jinpan_feature_range(40000, 60000).txt')
    jinpan_8 = pd.read_csv('data/jiaka_jinpan_feature_range(60000, 80000).txt')
    jinpan_10 = pd.read_csv('data/jiaka_jinpan_feature_range(80000, 100000).txt')
    jinpan_12 = pd.read_csv('data/jiaka_jinpan_feature_range(100000, 120000).txt')
    jinpan_14 = pd.read_csv('data/jiaka_jinpan_feature_range(120000, 140000).txt')
    jinpan_16 = pd.read_csv('data/jiaka_jinpan_feature_range(140000, 160000).txt')
    jinpan_18 = pd.read_csv('data/jiaka_jinpan_feature_range(160000, 180000).txt')
    jinpan_20 = pd.read_csv('data/jiaka_jinpan_feature_range(180000, 200000).txt')
    jinpan_22 = pd.read_csv('data/jiaka_jinpan_feature_range(200000, 220000).txt')

    frames = [jinpan_2, jinpan_4,jinpan_6,jinpan_8,jinpan_10,
              jinpan_12,jinpan_14,jinpan_16,jinpan_18,jinpan_20,jinpan_22]

    return pd.concat(frames)

def addr_concat():
    jinpan_0 = pd.read_csv('data/jiaka_addr_feature_range(0, 20000).csv')
    jinpan_2 = pd.read_csv('data/jiaka_addr_feature_range(20000, 40000).csv')
    jinpan_4 = pd.read_csv('data/jiaka_addr_feature_range(40000, 60000).csv')
    jinpan_6 = pd.read_csv('data/jiaka_addr_feature_range(60000, 80000).csv')
    jinpan_8 = pd.read_csv('data/jiaka_addr_feature_range(80000, 100000).csv')
    jinpan_10 = pd.read_csv('data/jiaka_addr_feature_range(100000, 120000).csv')
    jinpan_12 = pd.read_csv('data/jiaka_addr_feature_range(120000, 140000).csv')
    jinpan_14 = pd.read_csv('data/jiaka_addr_feature_range(140000, 160000).csv')
    jinpan_16 = pd.read_csv('data/jiaka_addr_feature_range(160000, 180000).csv')
    jinpan_18 = pd.read_csv('data/jiaka_addr_feature_range(180000, 200000).csv')
    jinpan_20 = pd.read_csv('data/jiaka_addr_feature_range(200000, 208161).csv')

    frames = [jinpan_0,jinpan_2, jinpan_4,jinpan_6,jinpan_8,jinpan_10,
              jinpan_12,jinpan_14,jinpan_16,jinpan_18,jinpan_20]

    return pd.concat(frames)


def extend_jin_addr_merge():
    # 纵向合并
    jinpan_df = jinpan_concat()

    extend_df = pd.read_csv('data/extend_info_feature.csv', sep='|')

    extend_clean_df = extend_clean(extend_df)

    addr_df = addr_concat()

    # jinpan_df = jinpan_df[['order_id','label']]

    print('jinapan', jinpan_df.shape)
    print('extend_info', extend_clean_df.shape)
    print('addr', addr_df.shape)

    # 横向合并
    all_df = pd.merge(extend_clean_df, jinpan_df, on='order_id')

    all_df = pd.merge(all_df, addr_df, on='order_id')

    print('mix all ', all_df.shape)

    all_df.to_csv('all_feature_1.csv')
    # all_df.to_csv('extend_info_feature_5w.csv')

def tele_ex_jin_addr_merge():
    odl_df = pd.read_csv('D:/develop/test/model_build/social/data/jiaka_social_feature_6w.csv')
    new_df = pd.read_csv('D:/develop/test/model_build/social/data/jiaka_social_comm_feature2018_test.csv')

    cat_features = [cont for cont in list(new_df.select_dtypes(
        include=['float64', 'int64']).columns) if 'one' not in cont]

    print(cat_features)

    new_df = new_df[cat_features]
    new_df.drop([  'loan_no', 'overdue_days', 'label'], axis=1, inplace=True)

    print('odl_df', odl_df.shape)
    print('new_df', new_df.shape)

    # 横向合并
    all_df = pd.merge(odl_df, new_df, on='order_id')

    print('mix all ', all_df.shape)

    all_df.to_csv('social_comm_feature_2018.csv',index=False)

def emer_merge():
    e3_df = pd.read_csv('data/social_emer_2019_test3.csv')

    e3_df.drop(['create_time','md5_num','m1_enev','first_overdue_days',
                 'product_id','slpd7','m1_times','m2','m3','m4','m5',
                'related_user_count','faith_break_users_count','faith_break_users_odd',
                'same_phone_ids_count','each_other_emer_ids_count',
                'related_emer_phone_count','related_moble_count','related_emer_phone_rate',
                'related_emer_phone_odd','related_degree','related_avg_degree','related_user_age_max',
                'related_user_age_min', 'related_user_age_mean', 'related_user_age_std', 'related_age_max_diff',
                'related_age_old_exist', 'same_emer_cheat_names_count', 'same_emer_unsame_names_count', 'same_emer_apprear_names_count',
                'same_emer_fruad_names_count'
                ], axis=1, inplace=True)
    e4_df = pd.read_csv('data/social_emer_2019_test4.csv')

    print('e3_df', e3_df.shape)
    print('e4_df', e4_df.shape)

    # 横向合并
    all_df = pd.merge(e4_df,e3_df, on='order_id',)

    print('mix all ', all_df.shape)

    all_df.to_csv('social_emer_2019_test6.csv', index=False)


def common_merge():
    emer_df = pd.read_csv('data/tqy_duotou_sub.csv')
    tqy_df = pd.read_csv('data/social_emer_2019_test3.csv')

    print('emer_df', emer_df.shape)
    print('tqy_df', tqy_df.shape)

    # 横向合并
    all_df = pd.merge(emer_df, tqy_df, on='md5_num',how='inner')

    print('all_df', all_df.shape)

    all_df.to_csv('data/钛旗云数据解析.csv', index=False)

def xinyan_clean(xinyan_df,xinyan_new_df):
    # print(xinyan_df.shape,xinyan_new_df.shape)
    order_id = xinyan_df['order_id']
    order_id = set(order_id)
    new_order_id = xinyan_new_df['order_id']
    new_order_id = set(new_order_id)

    # 取交集
    tmp = order_id&new_order_id

    # 新数据取交集
    xinyan_new_df = xinyan_new_df[xinyan_new_df['order_id'].isin(list(tmp))]


    # 老数据取逆
    xinyan_df = xinyan_df[~xinyan_df['order_id'].isin(list(tmp))]

    # print(xinyan_df.shape, xinyan_new_df.shape)

    frames = [xinyan_df,xinyan_new_df]

    new_df = pd.concat(frames)

    # print(new_df.shape)

    return new_df


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

    endtime = time.time()
    print(' cost time: ', endtime - starttime)

