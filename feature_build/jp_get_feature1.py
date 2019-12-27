import multiprocessing
import requests
import json
import pandas as pd
import numpy as np
import pymysql

#################################
# 最新接口

jinpan_url = "http://172.19.132.246:8023/getJinpanLoanFeatures?identityNo=%s&create_time=%s"


df_all = pd.read_csv("D:/特征提取/嘉卡/20190801095422tmp_jk.txt")
# df_all = df_all.head(8000)


print(df_all.shape)
print("唯一order_id",len(np.unique(df_all.order_id)))
# df_all.head(3)
print(df_all.shape)

print(df_all.head())



########
# tupan-test 02
# 47.101.206.54(弹性)
# 172.19.133.11(私有)


#  获取数据库连接
def getPymysql(ahost,auser,pw,Db,aport):
    # read_con = pymysql.connect(host='rm-uf6j98p717px3v8pj5o.mysql.rds.aliyuncs.com', user='tianshen',
    #                            password='tianshen', database='tianshen_syn_temp', port=3306, charset='utf8')

    read_con = pymysql.connect(host=ahost, user=auser,
                               password=pw, database=Db, port=aport, charset='utf8')

    return read_con

write_folder = 'D:/特征提取/嘉卡/data/'
#
def save_feature(range_index):
    sql = "select uo.create_time,ub.identity_no_ency from user_order uo " \
          "left join user_basics_info ub on ub.user_id = uo.user_id where merchant_id='11' and loan_order_no = '%s'"
    # 读数据
    read_con = getPymysql('139.196.75.48', 'root', 'zhouao.123',
                          'user_db', 8066)
    cursor = read_con.cursor()

    f = open(write_folder+"jiaka_jinpan_feature_%s.txt" %(str(range_index)),'w')
    # range_index = range(130,140)
    line_cnt = 0

    # 2542 231750373882732544 feature_size not eqal 1035

    for index in range_index:
        # index=2542
        # index=134 # 全空
        loan_no = df_all.loc[index, "loan_no"]
        create_time = df_all.loc[index, "create_time"]
        order_id = df_all.loc[index, "order_id"]
        label = df_all.loc[index, "label"]
        overdue_days = df_all.loc[index, "overdue_days"]


        try:

            cursor.execute(sql % (loan_no))
            re = cursor.fetchone()
            currDate = re[0]
            idNum = re[1]

            feat_title = ["order_id"]
            feat_value = [str(order_id)]

            feat_title.append('create_time')
            feat_value.append(create_time)

            feat_title.append('loan_no')
            feat_value.append(str(loan_no))

            feat_title.append('md5_num')
            feat_value.append(str(idNum))

            feat_title.append('label')
            feat_value.append(str(label))
            feat_title.append('overdue_days')
            feat_value.append(str(overdue_days))

            jinpan_request_url = jinpan_url % (idNum,currDate)
            # jinpan_request_url = "http://47.101.206.54:8023/getModelEncyLoanFeaturesTimeback?identityNo=9d186be94f3cc676f1a9199b21e58ed6&currDate=20190701"
            jinpan_response = requests.get(jinpan_request_url)
            features = json.loads(str(jinpan_response.content, 'utf-8')).get('result').get('features')
            # print(features.keys()

            # json.dump(features,open("/restore/working/litengfei/PaydayStandard_2/Data/jinpan_feature_eg/jinpan_feature_example.json","w"))


            for key,values in features.items():
                for feat,value in values.items():
                    if not isinstance(value,list) and not isinstance(value,dict):
                       feat_title.append(feat)
                       feat_value.append(value)
                    if isinstance(value,dict):
                       for var,value_str in value.items():
                           if not isinstance(value_str,list) and not isinstance(value_str,dict):
                              feat_title.append(var)
                              feat_value.append(value_str)

            # print(len(feat_title), len(feat_value), "\n")

            if len(feat_value) == 1040:
                print(index, order_id)
                line_cnt = line_cnt +1
                if line_cnt == 1:
                   f.write(','.join(feat_title)+'\n')
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
                else:
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
            else:
                print(index, order_id, "feature_size not eqal 1035",len(feat_value))

        except Exception as e:
            print(index, order_id,e)
            cursor.close()
            read_con.close()

            read_con = getPymysql('139.196.75.48', 'root', 'zhouao.123',
                                  'user_db', 8066)
            cursor = read_con.cursor()

    f.close()
    cursor.close()
    read_con.close()





# save_feature(range(525130,525140))
# len_df = len(df_all)
len_df = len(df_all)
minmax_inds = [k for k in range(100000, len_df, 20000)] + [len_df]
ls_inds = []
for k in range(len(minmax_inds) - 1):
    ls_inds.append(range(minmax_inds[k], minmax_inds[k + 1]))

print(ls_inds)


save_feature(range(80000, 100000))


# pool = multiprocessing.Pool(2)
# for iiinds in ls_inds:
#     pool.apply_async(save_feature, [iiinds,])
# # #
# pool.close()
# pool.join()



###############################

# print("\n合并数据\n")
#
# import os
#
# jp_feature_file_list = os.listdir(write_folder)
#
# jp_feature = pd.DataFrame()
# for jp_feature_file in jp_feature_file_list:
#     jp_feature_i = pd.read_csv(write_folder+jp_feature_file)
#     print(jp_feature_i.shape)
#     jp_feature = pd.concat([jp_feature,jp_feature_i],axis=0)
#
# print(jp_feature.shape)
# jp_feature.to_csv(write_folder + "a19_jinpan_feature.csv",index=False)






# [range(336000, 357000),
#  range(357000, 378000),
#  range(378000, 399000),
#  range(399000, 420000),
#  range(420000, 441000),
#


#  range(441000, 462000),
#  range(462000, 483000),
#  range(483000, 504000),



#  新机器跑这个
#  range(504000, 525000),
#  range(525000, 546000),
#  range(546000, 567000),
#  range(567000, 588000),
#  range(588000, 609000),
#  range(609000, 630000),
#  range(630000, 651000),
#  range(651000, 656458)]



