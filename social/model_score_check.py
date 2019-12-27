import pandas as pd
import numpy as np
import lightgbm as lgb
import math
# 计算和评估样本模型评分

model_lgb_pd7 = lgb.Booster(model_file='model/model_emer_online_pd7.lgb')
model_lgb_m1 = lgb.Booster(model_file='model/model_emer_online_m1_ever.lgb')
model_lgb_m2 = lgb.Booster(model_file='model/model_emer_online_m2.lgb')
model_lgb_m3 = lgb.Booster(model_file='model/model_emer_online_m3.lgb')
model_lgb_m4 = lgb.Booster(model_file='model/model_emer_online_m4.lgb')
model_lgb_m5 = lgb.Booster(model_file='model/model_emer_online_m5.lgb')

def model_score_compute():
    allData = pd.read_csv('data/social_emer_2019_online.csv',error_bad_lines=False )

    allData['overdueday'] = allData['m4']

    model_score_df = allData[['order_id', 'create_time', 'md5_num', 'm1_enev', 'first_overdue_days',
                  'product_id', 'slpd7', 'm1_times', 'm2', 'm3', 'm4', 'm5']]

    allData.drop(['order_id', 'create_time', 'md5_num', 'm1_enev', 'first_overdue_days',
                  'product_id', 'slpd7', 'm1_times', 'm2', 'm3', 'm4', 'm5'], axis=1, inplace=True)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    cont != 'overdueday']


    x_array = np.array(allData[cat_features])

    y_pred_pd7 = model_lgb_pd7.predict(x_array)
    y_pred_m1 = model_lgb_m1.predict(x_array)
    y_pred_m2 = model_lgb_m2.predict(x_array)
    y_pred_m3 = model_lgb_m3.predict(x_array)
    y_pred_m4 = model_lgb_m4.predict(x_array)
    y_pred_m5 = model_lgb_m5.predict(x_array)


    score_list_pd7 = []
    score_m_pd7 = []
    for s in y_pred_pd7:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s,score)
        m_s = score_map(score)
        score_m_pd7.append(m_s)
        score_list_pd7.append(score)

    score_list_m1 = []
    score_m_m1 = []
    for s in y_pred_m1:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m1.append(m_s)
        score_list_m1.append(score)

    score_list_m2 = []
    score_m_m2 = []
    for s in y_pred_m2:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m2.append(m_s)
        score_list_m2.append(score)

    score_list_m3 = []
    score_m_m3 = []
    for s in y_pred_m3:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m3.append(m_s)
        score_list_m3.append(score)

    score_list_m4 = []
    score_m_m4 = []
    for s in y_pred_m4:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m4.append(m_s)
        score_list_m4.append(score)

    score_list_m5 = []
    score_m_m5 = []
    for s in y_pred_m5:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m5.append(m_s)
        score_list_m5.append(score)

    # pred_df = pd.DataFrame(y_pred)
    score_df_pd7_m = pd.DataFrame(score_m_pd7)
    score_df_m1_m = pd.DataFrame(score_m_m1)
    score_df_m2_m = pd.DataFrame(score_m_m2)
    score_df_m3_m = pd.DataFrame(score_m_m3)
    score_df_m4_m = pd.DataFrame(score_m_m4)
    score_df_m5_m = pd.DataFrame(score_m_m5)

    score_df_pd7 = pd.DataFrame(score_list_pd7)
    score_df_m1 = pd.DataFrame(score_list_m1)
    score_df_m2 = pd.DataFrame(score_list_m2)
    score_df_m3 = pd.DataFrame(score_list_m3)
    score_df_m4 = pd.DataFrame(score_list_m4)
    score_df_m5 = pd.DataFrame(score_list_m5)

    new_df = pd.concat([model_score_df,
                        score_df_pd7,score_df_pd7_m,
                        score_df_m1,score_df_m1_m,
                        score_df_m2,score_df_m2_m,
                        score_df_m3,score_df_m3_m,
                        score_df_m4,score_df_m4_m,
                        score_df_m5,score_df_m5_m], axis=1, ignore_index=True)
    print(model_score_df.shape, new_df.shape)

    p_col = ['order_id', 'create_time', 'md5_num', 'm1_enev', 'first_overdue_days',
                  'product_id', 'slpd7', 'm1_times', 'm2', 'm3', 'm4', 'm5',
             'model_score_pd7','model_score_pd7_m',
             'model_score_m1','model_score_m1_m',
             'model_score_m2','model_score_m2_m',
             'model_score_m3','model_score_m3_m',
             'model_score_m4','model_score_m4_m',
             'model_score_m5','model_score_m5_m']

    new_df.columns=p_col
    new_df.to_csv('score_data/all_online_model_score_map.csv')

def score_map(x):
    max_score = 900
    min_score = 300

    dis = 150

    # 映射结果
    y = round(float(dis - (dis * (x - min_score) / (max_score-min_score))),3)
    return y

def model_score_filter():
    pass


if __name__ == '__main__':
    model_score_compute()