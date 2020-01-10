import pandas as pd
import numpy as np
import lightgbm as lgb
import math
# 计算和评估样本模型评分

model_lgb_m5 = lgb.Booster(model_file='model/model_afu.lgb')

def model_score_compute():
    allData = pd.read_excel('data/zcaf_data.xlsx', encoding='utf8')

    # allData['overdueday'] = allData['m4']

    model_score_df = allData[['order_id', 'create_time', 'md5_mobile',
                              'name','product_name','label']]

    allData.drop(['order_id', 'create_time', 'md5_mobile', 'name', 'overdue_days',
                  'product_name', 'loan', '身份证', '手机号', 'label'], axis=1, inplace=True)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if
                    cont != 'overdueday']


    x_array = np.array(allData[cat_features])

    y_pred_m5 = model_lgb_m5.predict(x_array)

    score_list_m5 = []
    score_m_m5 = []
    for s in y_pred_m5:
        score = round(math.log2((1 - s) / s) * 50 + 450, 2)
        print(s, score)
        m_s = score_map(score)
        score_m_m5.append(m_s)
        score_list_m5.append(score)

    # score_df_m5_m = pd.DataFrame(score_m_m5)
    score_df_m5 = pd.DataFrame(score_list_m5)

    new_df = pd.concat([model_score_df,score_df_m5], axis=1, ignore_index=True)
    print(model_score_df.shape, new_df.shape)

    p_col = ['order_id', 'create_time', 'md5_mobile',
                              'name','product_name','label','model_score']

    new_df.columns=p_col
    new_df.to_csv('model/afu_model_score.csv')

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