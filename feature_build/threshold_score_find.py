
import pandas as pd

# 根据通过率计算门槛分
def thresh_score_check(df,persent):
    # 过滤模型号
    df = model_type_split(df)

    df = df.sort_values(by='obscure_score_inta27', ascending=False)

    df = df.dropna(axis=0, how='any')
    df.fillna(0)
    # print(df)
    rec_num = df.shape[0]

    good = df.head(int(rec_num/(1/persent)))

    last = good.iloc[-1]

    print(last)

    # good.to_csv('filter_score.csv')

# 通过第一道门槛分确认通过样本
def thresh_score_filter(df,score):
    # 过滤模型号
    # df = model_type_split(df)

    df = df.sort_values(by='score', ascending=False)

    rec_num = df.shape[0]

    filter_df = score_threahold_split(df,score)

    print(filter_df.shape)

    # filter_df.to_csv('filter_first_threadhold_order.csv')

# 不同通过率下的 第二道门槛分确认
def thresh_sec_score_filter(df,score, approve_rate):
    # 过滤模型号
    # df = model_type_split(df)

    df = df.sort_values(by='offline_score', ascending=False)

    rec_num = df.shape[0]
    top_num = int(rec_num * approve_rate)

    good = df.head(top_num)
    print('all', rec_num, 'top_num', top_num,'top shape',good.shape)

    last = good.iloc[-1]

    print(last)
    # filter_df.to_csv('filter_first_threadhold_order.csv')

def score_threahold_split(allData,threahold_score):
    allData['split1'] = allData['score'].apply(
        lambda x: 'all' if
                            x >= threahold_score
        else 'ear')


    for name, group in allData.groupby('split1'):
        print(name, group.shape)
        if name is 'all':
            all = group
        else:
            train = group

    all.drop(['split1'], axis=1, inplace=True)

    return all

def new_old_xinyan_model_comp():
    old_xinyan = pd.read_csv('filter_first_threadhold_order.csv')

    new_xinyan = pd.read_csv('threahold_offline_score.csv')

    print(old_xinyan.describe())
    print(new_xinyan.describe())

    comp_df = pd.merge(old_xinyan,new_xinyan,on='order_id',how='inner')

    print(comp_df.shape)

def model_type_split(allData):
    allData['split1'] = allData['model_type'].apply(
        lambda x: 'all' if
                            x == 'model_tele'
        else 'ear')


    for name, group in allData.groupby('split1'):
        print(name, group.shape)
        if name is 'all':
            all = group
        else:
            train = group

    all.drop(['split1'], axis=1, inplace=True)

    return all

def model_type_split_pro(allData):
    allData['split1'] = allData['model_flow_name'].apply(
        lambda x: 'all' if
                            x == 'jk_mx_sd_A27B11_0005'
        else 'ear')


    for name, group in allData.groupby('split1'):
        # print(name, group.shape)
        if name is 'all':
            all = group
        else:
            train = group

    all.drop(['split1'], axis=1, inplace=True)

    return all

def sec_threahold_score_oot_check(score):
    oot_0607 = pd.read_csv('oot_0413_0515_offline_score.csv')
    top = sec_score_split(oot_0607, score)

    all_pd7_rate = oot_0607.label.sum() / oot_0607.shape[0]
    top_pd7_rate = top.label.sum() / top.shape[0]

    print('all',oot_0607.shape,'approve',top.shape)

    print('score',score,'all pd7 ',all_pd7_rate,'approve pd7 ',top_pd7_rate)

# 对oot数据不同模型进行区分
def sec_threahold_score_oot_model_type_check(score):
    label_0607 = pd.read_csv('20190822141101tmp_jk_0608.txt')
    label_0607 = model_type_split_pro(label_0607)
    label_0607.drop(['label'], axis=1, inplace=True)

    oot_0607 = pd.read_csv('oot_offline_score.csv')

    oot_0607  = pd.merge(label_0607,oot_0607,on='order_id',how='inner')

    top = sec_score_split(oot_0607, score)

    all_pd7_rate = oot_0607.label.sum() / oot_0607.shape[0]
    top_pd7_rate = top.label.sum() / top.shape[0]

    print('all',oot_0607.shape,'approve',top.shape)

    print('score',score,'all pd7 ',all_pd7_rate,'approve pd7 ',top_pd7_rate)

# 门槛分切分
def sec_score_split(allData,score):
    allData['split1'] = allData['offline_score'].apply(
        lambda x: 'all' if
                            x >= score
        else 'ear')


    for name, group in allData.groupby('split1'):
        # print(name, group.shape)
        if name is 'all':
            all = group
        else:
            train = group

    all.drop(['split1'], axis=1, inplace=True)

    return all

if __name__ == '__main__':
    df = pd.read_csv('20190904190816tmp_jk_071516.csv')
    #总体通过率10%  第一道 A27门槛分 按50%的通过率
    thresh_score_check(df, 0.5)

    df_xinyan = pd.read_csv('071516_xinyan_A27_feature.txt')
    # 1  通过第一道门槛分确认通过样本
    thresh_score_filter(df_xinyan, 626.31)

    # 2  不同通过率下的 第二道门槛分确认
    # threahold_offline_score = pd.read_csv('threahold_offline_score.csv')
    # thresh_sec_score_filter(threahold_offline_score,626.31,0.20)

    # 查看 新模型与老模型对同一批样本给出的分数差异
    # new_old_xinyan_model_comp()

    # 3  不同门槛分下的贷后情况
    # sec_threahold_score_oot_check(715.87)

    # score = 600
    # while score < 850:
    #     score = score + 10
    #     sec_threahold_score_oot_check(score)
    #
    # sec_threahold_score_oot_model_type_check(718.7)

    # score = 600
    # while score < 850:
    #     score = score + 10
    #     sec_threahold_score_oot_model_type_check(score)




