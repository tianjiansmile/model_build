
from scipy.stats import wasserstein_distance
# import pyemd
import numpy as np
import pandas as pd
import datetime
import math

def getFeatureFunction(feat1, feat2, func='minus'):
    if func == 'minus':
        try:
            return feat1 - feat2
        except:
            return -1111
    elif func == 'divide':
        try:
            return feat1 / feat2
        except:
            return -1111

def getDeriveFeature(data):
    derive_dict = dict()
    print(data.shape)
    data['主叫联系人黑名单数_rate'] = getFeatureFunction(data['主叫联系人黑名单数'],
                                                                             data['主叫联系人数'], 'divide')
    data['主叫联系人逾期个数_rate'] = getFeatureFunction(data['主叫联系人逾期个数'],
                                                       data['主叫联系人数'], 'divide')

    data['夜间通话次数_avg'] = getFeatureFunction(data['夜间通话次数'],
                                                data['夜间通话人数'], 'divide')
    data['与虚拟号码通话次数_avg'] = getFeatureFunction(data['与虚拟号码通话次数'],
                                                data['与虚拟号码通话人数'], 'divide')

    data['异地通话次数_avg'] = getFeatureFunction(data['异地通话次数'],
                                            data['异地通话人数'], 'divide')
    data['与澳门通话人数_avg'] = getFeatureFunction(data['与澳门通话次数'],
                                               data['与澳门通话人数'], 'divide')

    data['与银行或同行通话次数_avg'] = getFeatureFunction(data['与银行或同行通话总次数'],
                                            data['与银行或同行通话总人数'], 'divide')

    print(data.shape)

    data.to_excel('data/zcaf_data2.xlsx',index=False)



if __name__ == '__main__':
    df = allData = pd.read_excel('data/zcaf_data.xlsx', encoding='utf8')
    # df.drop(['order_id', 'create_time', 'md5_mobile', 'name', 'overdue_days',
    #               'product_name', 'loan', '身份证', '手机号', 'label'], axis=1, inplace=True)
    # feature_name = [cont for cont in list(allData.select_dtypes(
    #     include=['float64', 'int64']).columns) if
    #                 # 'high' not in cont and
    #
    #                 cont != 'overdueday']
    # df_derive_res = pd.DataFrame(list(df.apply(lambda x: getDeriveFeature(x)[1], axis=1)), columns=feature_name)
    getDeriveFeature(df)