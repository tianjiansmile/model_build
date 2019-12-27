import pandas as pd

mz_key=['cash_loan_15d',
'auth_contactnum_ratio_90d',
'auth_intimatenum_ratio_30d',
'org_count',
'datacoverge_90d',
'black_intimate_indirectnum_ratio_180d',
'match_score',
'black_indirect_peernum_ratio_90d',
'black_indirectnum_ratio_180d',
'black_intimate_indirect_peernum_ratio_30d',
'black_indirectnum_ratio_30d',
'other_count',
'auth_indirectnum_ratio_30d',
'cash_loan_30d',
'cash_loan_90d',
'consumstage_180d',
'blacklist_record_overdue_count',
'gray_record_overdue_count']

ym_key=['appStability180d',
'appStability7d',
'appStability90d',
'car180d',
'car7d',
'car90d',
'cityFreq',
'cityRec',
'countryFreq',
'countryRec',
'creditScore',
'deviceBrand',
'deviceOs',
'devicePrice',
'deviceRank',
'education180d',
'education7d',
'education90d',
'entertainment180d',
'entertainment7d',
'entertainment90d',
'finance180d',
'finance7d',
'finance90d',
'game180d',
'game7d',
'game90d',
'health180d',
'health7d',
'health90d',
'ip90d',
'launchDay',
'loan180d',
'loan7d',
'loan90d',
'property180d',
'property7d',
'property90d',
'provinceFreq',
'provinceRec',
'reading180d',
'reading7d',
'reading90d',
'service180d',
'service7d',
'service90d',
'shopping180d',
'shopping7d',
'shopping90d',
'sns180d',
'sns7d',
'sns90d',
'tail180d',
'tail7d',
'tail90d',
'tools180d',
'tools7d',
'tools90d',
'top180d',
'top7d',
'top90d',
'travel180d',
'travel7d',
'travel90d',
'woman180d',
'woman7d',
'woman90d']
hj_key=['loan_account','loan_amt','outstand_count','loan_bal','overdue_count','overdue_amt','overdue_more_count','overdue_more_amt','generation_count','generation_amount']
bj_key=['houmou_score']
id_key = ['order_id']
mz_key.extend(ym_key)
mz_key.extend(hj_key)
mz_key.extend(bj_key)
mz_key.extend(id_key)

map_dict = {"app_stability_180d":"appStability180d","app_stability_7d":"appStability7d","app_stability_90d":"appStability90d","car_180d":"car180d","car_7d":"car7d","car_90d":"car90d","cid":"cid","city_freq":"cityFreq","city_rec":"cityRec","code":"code","country_freq":"countryFreq","country_rec":"countryRec","create_time":"createTime","credit_score":"creditScore","cust_id":"custId","device_brand":"deviceBrand","device_os":"deviceOs","device_price":"devicePrice","device_rank":"deviceRank","education_180d":"education180d","education_7d":"education7d","education_90d":"education90d","entertainment_180d":"entertainment180d","entertainment_7d":"entertainment7d","entertainment_90d":"entertainment90d","finance_180d":"finance180d","finance_7d":"finance7d","finance_90d":"finance90d","game_180d":"game180d","game_7d":"game7d","game_90d":"game90d","health_180d":"health180d","health_7d":"health7d","health_90d":"health90d","id":"id","ip_90d":"ip90d","is_history":"isHistory","launch_day":"launchDay","loan_180d":"loan180d","loan_7d":"loan7d","loan_90d":"loan90d","mobile":"mobile","property_180d":"property180d","property_7d":"property7d","property_90d":"property90d","province_freq":"provinceFreq","province_rec":"provinceRec","reading_180d":"reading180d","reading_7d":"reading7d","reading_90d":"reading90d","service_180d":"service180d","service_7d":"service7d","service_90d":"service90d","shopping_180d":"shopping180d","shopping_7d":"shopping7d","shopping_90d":"shopping90d","sns_180d":"sns180d","sns_7d":"sns7d","sns_90d":"sns90d","source_app":"sourceApp","status":"status","tail_180d":"tail180d","tail_7d":"tail7d","tail_90d":"tail90d","tools_180d":"tools180d","tools_7d":"tools7d","tools_90d":"tools90d","top_180d":"top180d","top_7d":"top7d","top_90d":"top90d","travel_180d":"travel180d","travel_7d":"travel7d","travel_90d":"travel90d","update_time":"updateTime","woman_180d":"woman180d","woman_7d":"woman7d","woman_90d":"woman90d"}

def data_split(allData):
    allData['split1'] = allData['create_time'].apply(
        lambda x: 'all' if  x >= '2019-05-19'
        else 'ear')

    for name, group in allData.groupby('split1'):
        print(name, group.shape)
        if name is 'all':
            all = group
        else:
            train = group

    all.drop(['split1'], axis=1, inplace=True)

    return all

def three_feature_merge():
    all_df = pd.read_csv('/restore/working/panjin/20190806113651tmp_jk.txt')

    all_df = data_split(all_df)


    mz_df = pd.read_csv('/restore/working/youxin/nwd_jk/feature/feature/feature_mz.txt',sep='\t')
    mz_df['loan_no'] = mz_df['camera_id_var']

    mz_df.drop(['camera_id_var'], axis=1,
                      inplace=True)

    # mz_df = mz_df[mz_key]

    # 横向合并
    all_df = pd.merge(all_df, mz_df, on='loan_no', how='inner')

    print('mix 1', all_df.shape)


    # # 友盟、互金 冰鉴
    three_all_df = pd.read_csv('/restore/working/youxin/nwd_jk/label/jk_three_data_label.txt', sep='\t')

    three_all_df['loan_no'] = three_all_df['camera_id_var']

    three_all_df.drop(['camera_id_var'], axis=1,
                      inplace=True)

    three_all_df.rename(columns=map_dict, inplace=True)
    #
    # # 横向合并
    all_df = pd.merge(all_df, three_all_df, on='loan_no', how='inner')
    #
    print('mix 2 ', all_df.shape)

    all_df = all_df[mz_key]
    print('mix 2 clear', all_df.shape)
    #
    # # # 将不参与训练的特征数据删除
    # all_df.drop(['create_time', 'md5_num','loan_no', 'overdue_days', 'model_flow_name', 'model_type',
    #              'order_type', 'product_name', 'sourcechannle'], axis=1,
    #             inplace=True)
    #
    print(all_df.head(20))
    #
    all_df.to_csv('data/three_0512_0622.csv')


if '__main__' == __name__:
        three_feature_merge()