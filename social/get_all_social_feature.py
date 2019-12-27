import multiprocessing
import requests
import json
import pandas as pd
import numpy as np
import pymysql

#################################
# 最新接口

jinpan_url = "http://172.19.133.30:8127/getSocialFeaturesBins?identityNo=%s"
# jinpan_url = "http://127.0.0.1:5000/getSocialFeatures?identityNo=%s&create_time=%s"


df_all = pd.read_csv("data/mobile_idum_whole.txt")
# df_all = pd.read_csv("data/2018tmp_jk.txt")
# df_all = df_all.head(100000)


print(df_all.shape)
print("唯一md5_num",len(np.unique(df_all.md5_num)))
# df_all.head(3)
print(df_all.shape)

print(df_all.head())

write_folder = 'data/'

def save_feature(range_index):

    f = open(write_folder+"whole_fea_bins.csv" ,'w')
    # range_index = range(130,140)
    line_cnt = 0

    # 2542 231750373882732544 feature_size not eqal 1035

    for index in range_index:
        # index=2542
        # index=134 # 全空
        # loan_no = df_all.loc[index, "loan_no"]
        # create_time = df_all.loc[index, "create_time"]
        # order_id = df_all.loc[index, "order_id"]
        # m1_enev = df_all.loc[index, "m1_enev"]
        # first_overdue_days = df_all.loc[index, "first_overdue_days"]
        # product_id = df_all.loc[index, "product_id"]
        md5_num = df_all.loc[index, "md5_num"]
        # slpd7 = df_all.loc[index, "slpd7"]


        try:


            feat_title = ["md5_num"]
            feat_value = [str(md5_num)]
            #
            # feat_title.append('create_time')
            # feat_value.append(create_time)

            # feat_title.append('md5_num')
            # feat_value.append(str(md5_num))

            # feat_title.append('m1_enev')
            # feat_value.append(str(m1_enev))
            # feat_title.append('first_overdue_days')
            # feat_value.append(str(first_overdue_days))
            #
            # feat_title.append('product_id')
            # feat_value.append(str(product_id))
            # feat_title.append('slpd7')
            # feat_value.append(str(slpd7))

            # print(create_time)

            # create_time = create_time + ' 00:00:00'
            # create_time = create_time + ' 00:00:00'

            jinpan_request_url = jinpan_url % (md5_num)
            print(jinpan_request_url)
            # jinpan_request_url = "http://47.101.206.54:8023/getModelEncyLoanFeaturesTimeback?identityNo=9d186be94f3cc676f1a9199b21e58ed6&currDate=20190701"
            jinpan_response = requests.get(jinpan_request_url)
            features = json.loads(str(jinpan_response.content, 'utf-8')).get('result')
            # print(features.keys()

            # json.dump(features,open("/restore/working/litengfei/PaydayStandard_2/Data/jinpan_feature_eg/jinpan_feature_example.json","w"))


            for feat,value in features.items():
                # if not isinstance(value,list) and not isinstance(value,dict):
                   feat_title.append(feat)
                   feat_value.append(value)
                # if isinstance(value,dict):
                #    for var,value_str in value.items():
                #        if not isinstance(value_str,list) and not isinstance(value_str,dict):
                #           feat_title.append(var)
                #           feat_value.append(value_str)

            # print(len(feat_title), len(feat_value), "\n")

            if len(feat_value) == 737:
                print(index, md5_num)
                line_cnt = line_cnt +1
                if line_cnt == 1:
                   f.write(','.join(feat_title)+'\n')
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
                else:
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
            else:
                print(index, md5_num, "feature_size not eqal ",len(feat_value))

        except:
            # raise  Exception
            print(index, md5_num)


save_feature(range(0, 20000000))