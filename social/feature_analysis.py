import pandas as pd
import matplotlib.pyplot as plt

def plot_check(df ):
    df = df.set_index('date')
    # df['int_count'].plot()
    # df['pdl_count'].plot()
    # df['sum_count'].plot()
    # plt.show()
    df.plot()
    plt.show()

def date_map(x):
    month = str(x)[5:7]

    return month

def data_check():
    # df = pd.read_csv('data/pdl_socal_fea.csv')

    df = pd.read_csv('data/jiaka_social_181012_two_range(0, 50000).csv')

    hit = df['SN_two_step_degree'].count()/df.shape[0]

    hit2 = df['SN_one_step_age_max'].count()/df.shape[0]
    #
    one_degree_mean = df['SN_one_step_degree'].mean()

    one_degree_high_mean = df['SN_one_step_high_degree'].mean()
    #
    two_degree_mean = df['SN_two_step_degree'].mean()

    two_degree_high_mean = df['SN_two_step_high_degree'].mean()

    print(df.shape)

    df['date'] = df['create_time'].map(date_map)

    # degree = df[['date','SN_two_step_degree','SN_two_step_high_degree']]

    degree = df[['date', 'SN_one_step_degree', 'SN_two_step_high_degree']]

    print('degree ', one_degree_mean, one_degree_high_mean, two_degree_mean, two_degree_high_mean)

    col = ['SN_one_step_call_len_mean','SN_one_step_high_call_len_mean','SN_one_step_call_time_mean','SN_one_step_high_call_time_mean',
           'SN_two_step_call_len_mean','SN_two_step_high_call_len_mean','SN_two_step_call_time_mean','SN_two_step_high_call_time_mean']
    for k in col:
        mean = df[k].mean()
        median = df[k].median()
        print(k + ' mean', mean)
        print(k+' median', median)

    # plot_check(overdue)

def data_loan_check():
    # df = pd.read_csv('data/pdl_socal_fea.csv')

    df = pd.read_csv('data/jiaka_social_181012_two_range(0, 50000).csv')


    print(df.shape)

    df['date'] = df['create_time'].map(date_map)

    apply = df[['date', 'SN_one_step_apply_sum_all_mean', 'SN_two_step_apply_sum_all_mean']]

    one_apply_mean = df['SN_one_step_apply_sum_all_mean'].mean()

    two_apply_mean = df['SN_two_step_apply_sum_all_mean'].mean()

    print('apply', one_apply_mean, two_apply_mean)

    overdue = df[['date', 'SN_one_step_overdue_sum_all_mean', 'SN_two_step_overdue_sum_all_mean']]

    one_overdue_mean = df['SN_one_step_overdue_sum_all_mean'].mean()

    two_overdue_mean = df['SN_two_step_overdue_sum_all_mean'].mean()

    print('overdue', one_overdue_mean, two_overdue_mean)

    one_overdue_avg = df['SN_one_step_overdue_sum_all_avg'].mean()

    two_overdue_avg = df['SN_two_step_overdue_sum_all_avg'].mean()

    print('overdue avg', one_overdue_avg, two_overdue_avg)

    one_approve_avg = df['SN_one_step_approve_sum_all_avg'].mean()

    one_high_approve_avg = df['SN_one_step_high_approve_sum_all_avg'].mean()

    two_approve_avg = df['SN_two_step_approve_sum_all_avg'].mean()

    two_high_approve_avg = df['SN_two_step_high_approve_sum_all_avg'].mean()

    print('approve avg', one_approve_avg, one_high_approve_avg,
          two_approve_avg, two_high_approve_avg)

    col = [
        'SN_one_step_apply_pdl_all_mean','SN_one_step_high_apply_pdl_all_mean',
           'SN_two_step_apply_pdl_all_mean', 'SN_two_step_high_apply_pdl_all_mean',
           'SN_one_step_approve_pdl_all_avg', 'SN_one_step_high_approve_pdl_all_avg',
           'SN_two_step_approve_pdl_all_avg', 'SN_two_step_high_approve_pdl_all_avg',
        'SN_one_step_reject_pdl_all_avg', 'SN_one_step_high_reject_pdl_all_avg',
        'SN_two_step_reject_pdl_all_avg', 'SN_two_step_high_reject_pdl_all_avg',
        #
        # 'SN_one_step_overdue_sum_all_mean', 'SN_one_step_high_overdue_sum_all_mean',
        # 'SN_two_step_overdue_sum_all_mean', 'SN_two_step_high_overdue_sum_all_mean',
        #
        'SN_one_step_overdue_pdl_all_avg', 'SN_one_step_high_overdue_pdl_all_avg',
        'SN_two_step_overdue_pdl_all_avg', 'SN_two_step_high_overdue_pdl_all_avg',
           ]
    for k in col:
        mean = df[k].mean()
        median = df[k].median()
        print(k + ' mean', mean)
        # print(k + ' median', median)

    # plot_check(overdue)

def oot_train_split(allData):
    allData['split'] = allData['create_time'].apply(lambda x: 'oot' if x <= '2018-11-27' else 'train')
    groupby = allData.groupby(allData['split'])
    for name, group in allData.groupby('split'):
        print(name,group.shape)
        if name is 'oot':
            oot = group
        else:
            train = group

    return oot

def concat_data():
    df_2019 = pd.read_csv('data/jiaka_social_feature2019_6w.csv')
    df = pd.read_csv('data/jiaka_social_feature_6w.csv')

    df = oot_train_split(df)

    print(df['create_time'].unique())

    print('df_2019',df_2019.shape)

    frames = [df,df_2019]

    new = pd.concat(frames)
    print(new.shape)
    new.to_csv('jiaka_social_fea.csv',encoding='utf-8',index=False)

def data_split(allData, product_id):
    allData['split1'] = allData['product_id'].apply(
        lambda x: 'oot' if x == product_id else 'train')
    for name, group in allData.groupby('split1'):
        if name is 'oot':
            oot1 = group

    oot1.drop(['split1'], axis=1, inplace=True)
    return oot1

def sample_filter():
    sample_df = pd.read_csv('D:/三方数据测试/int_20w_new.csv', encoding='utf8')
    sample_df['split1'] = sample_df['product_name'].apply(
        lambda x: 'oot' if x == '嘉卡' or x == '嘉优贷' else 'train')
    for name, group in sample_df.groupby('split1'):
        if name is 'oot':
            oot1 = group

    oot1.drop(['split1'], axis=1, inplace=True)

    print(oot1.shape)

    oot1['split1'] = oot1['create_time'].apply(
        lambda x: 'oot' if str(x) >= '20190201' else 'train')
    for name, group in sample_df.groupby('split1'):
        if name is 'oot':
            oot2 = group

    oot2.drop(['split1'], axis=1, inplace=True)
    print(oot2.shape)
    return oot1



if __name__ == '__main__':
    # data_check()
    # data_loan_check()

    # concat_data()

    sample_filter()

