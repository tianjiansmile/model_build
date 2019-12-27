#!/usr/bin/python
#-*- coding:utf-8 -*-

import json
import pymysql
import pandas as pd
from datetime import datetime
from collections import Counter
###从数据库拉数据
##########从数据库直接拉数据对变量

get_feature=r'test_one_feature.csv'
# get_feature=r'/restore/working/tianjian/jiaka/data/0607_online_feature.txt'
'''
connection = pymysql.connect(host='101.132.124.114',
                             user='root',
                             password='mysql',
                             db='risk',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
'''

connection = pymysql.connect(host='172.19.132.245',
                             user='root',
                             password='slp@2018by_ljy',
                             db='pdl',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor=connection.cursor()
############下面要改时间参数
sql='''SELECT  
loan_order_no as loan_order_no, 
UNCOMPRESS(exec_info) as data,
create_time as create_time
FROM instalment.risk_riskorderexecinfo 
WHERE create_time>'2019-09-06 10:33:01' AND create_time<='2019-09-06 11:27:41'
'''

l=[]
cursor.execute(sql)
tele_feature = open(get_feature,'w',encoding='utf-8')
f_list=[]
count=0
n=0
for row in cursor:
    #print(row)
    count+=1
    if count>1000000:
        break
    #order_id=row['loan_order_no'].split('/')[-1]
    order_id = row['loan_order_no']
    create_time=datetime.strftime(row['create_time'],'%Y-%m-%d %H:%M:%S')
    json_data=json.loads(row['data'])
    #print(json_data.keys(),'ssssssssssssss')
    try:
        data_feature = json_data['feature']
    except:
        continue
    try:
        data_obscure_score = json_data['score']['obscure_score']
    except:
        data_obscure_score = 'NaN'
    try:
        data_score = json_data['score']['score']
    except:
        data_score='NaN'

    title = []
    feature = []
    for key, value in data_feature.items():
        if type(value) != list:
            title.append(key)
            feature.append(value)
        else:
            for i in range(len(value)):
                title.append(key + '_' + str(i))
                feature.append(value[i])
    title.append('score')
    title.append('obscure_score')
    feature.append(data_score)
    feature.append(data_obscure_score)
    # print(len(feature))
    # f_list.append(len(feature))

    f_list.append(len(feature))
    print(order_id, len(feature), Counter(f_list))
    if len(feature) != 5171:
        continue
    n += 1
    split_str = '|'
    if n == 1:
        tele_feature.write('order_id' + split_str + split_str.join(title) + '\n')
        tele_feature.write(order_id + split_str + split_str.join(map(lambda x: str(x), feature)) + '\n')
    else:
        tele_feature.write(order_id + split_str + split_str.join(map(lambda x: str(x), feature)) + '\n')
tele_feature.close()
