import pandas as pd

# predData = pd.read_csv('extend_info_feature_test.csv',sep='|',
#                                    encoding='utf-8')
#
# print(predData['EI_tongdundata_hitRules_hitRules_3个月内手机在多个平台进行借款_score'],
#       predData['EI_tongdundata_hitRules_hitRules_3个月内设备在多个平台进行借款_score'],
#       predData['EI_tongdundata_hitRules_hitRules_7天内申请人手机号在多个平台申请借款_score'],
#       predData['EI_tongdundata_hitRules_hitRules_1个月内申请人手机号在多个平台申请借款_score'],)
#
# print(predData.shape)

def order_loan_map(catch,jk):
    order_loan = jk[['order_id','loan_no']]

    # 字段转字典
    order_loan_dict = order_loan.set_index("loan_no").to_dict()['order_id']

    order_id_list = []
    for id in catch:
        if order_loan_dict.get(id):
            order_id_list.append(order_loan_dict.get(id))

    return order_id_list



def wrong_data_handle():
    offline = pd.read_csv('0607_all_feature_pro.csv')
    off_order_id = offline[['order_id','label']]

    order_id =  offline['order_id']

    data_output = pd.read_csv('tmp_data/data_output.txt')
    out_order_id = data_output[['order_id','label']]

    republish_2019 = pd.read_excel('tmp_data/20190101-20190828期间内进件但退标的明细(1)(1).xlsx')
    republish_2018 = pd.read_excel('tmp_data/2018年进件但退标(2).xlsx')

    jk = pd.read_csv('/home/tianjian/jiaka/data/20190801095422tmp_jk.txt')
    loan_no = jk['loan_no'].tolist()

    camera_id = republish_2019['camera_id'].tolist()
    camera_id1 = republish_2018['camera_id'].tolist()

    print('delete before ', offline.shape)



    # 需要删除的loan
    catch = list(set(camera_id).intersection(set(loan_no)))
    catch1 = list(set(camera_id1).intersection(set(loan_no)))
    catch = catch.extend(catch1)

    print(len(catch))

    order_id_delete = order_loan_map(catch,jk)

    # 删除指定行
    offline = offline[~offline['order_id'].isin(order_id_delete)]

    offline = offline.reset_index(drop=True)

    print('delete loan', len(catch),'delete order_id', len(order_id_delete), 'after ', offline.shape)

    # 字段转字典
    off_order_id_dict = off_order_id.set_index("order_id").to_dict()['label']
    out_order_id_dict = out_order_id.set_index("order_id").to_dict()['label']

    print('off_order_id_dict',len(off_order_id_dict),'out_order_id_dict',len(out_order_id_dict))

    for off in off_order_id_dict:
        if out_order_id_dict.get(off):
            off_order_id_dict[off] = out_order_id_dict.get(off)
            print(off,off_order_id_dict[off],out_order_id_dict.get(off))

    off_order = pd.DataFrame.from_dict(off_order_id_dict, orient='index',columns=['label_fix'])

    off_order = off_order.reset_index().rename(columns={"index": "order_id"})
    # merge
    new = pd.merge(offline, off_order, on='order_id',how='inner')

    new.to_csv('0607_all_feature_pro_clean.csv')

    print()




