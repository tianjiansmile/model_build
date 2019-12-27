# -*- coding: utf-8 -*-

import os
import requests
import gzip
import json
import queue
import threading
import sys
import queue
import threading
import sys
import datetime
import re

num_worker_threads = 10

# filepath = r'/restore/working/tianjian/data/jiaka/report/tmp_jk_0608'
filepath = 'D:/develop/test/model_build/feature_build/file'
files = os.listdir(filepath)

url='http://172.19.132.221:8595/riskcontrol/risk/'
# url='http://172.19.132.221:8197/riskcontrol/risk/'

# fwriter_path = "/restore/working/tianjian/jiaka/data/test_all_feature.txt"
fwriter_path = "test_one_feature.csv"
fwriter = open(fwriter_path,mode="w")
fwriter.write("order_id,score,error\n")
q = queue.Queue()
start=datetime.datetime.now()
def post_scored(file):
    
    order_id=file.split('/')[-1].split('.')[0]
    
    json_data={}
    json_data['loanOrderNo']=order_id
    json_data['merchantNo']='100'
    #json_data['merchantNo']='100'
    json_data['modelNo']=None
    json_data['productNo']=10011
    json_data['uid']='123456'
    
    try:
       file_content = gzip.open(file, 'rt', encoding='utf-8').read()
    except:
       file_content=open(file,encoding='utf-8').read()
    
    #rep=re.compile(r'{(.+)}')
    #file_content=rep.findall(file_content)[0]
    try:
       request_json = json.loads(file_content)
    except:
       print('file %s can not parse'%file)
       return None
    #print(request_json.keys())
    
    
    #json_data['modelNo'] = 'DR_jxl_sd_1.0'
    json_data['data']=request_json
    #print(json_data['modelNo'])
    #print(json_data.keys())
    

    #file_content = json.dumps(request_json, ensure_ascii=False).encode('utf-8')  #2019-04-17 14:16:00
    file_content = json.dumps(json_data, ensure_ascii=False).encode('utf-8')  #2019-02-27 14:54:00
    try:
        r_new = requests.post(url, data=file_content)
        #print('new:', r_new.status_code, r_new.content, len(r_new.content))
        print('new:', r_new.status_code, r_new.content, len(r_new.content),file)
        fwriter.write(",".join(list(map(lambda x: str(x),[order_id, json.loads(r_new.content)["score"], json.loads(r_new.content)["return_message"]]))) + "\n")
        #f.write(str(json.loads(r_new.content.decode('utf-8'))['score'])+','+file.split('/')[-1].split('.')[0].split('_')[1]+'\n')
    except Exception as e:
        raise  Exception
        print('model calu error')




def worker():
    while True:
        item = q.get()
        post_scored(item)
        q.task_done()

for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()


count=0
now1=datetime.datetime.now()
for i in files:
    file_path_name = os.path.join(filepath, i)
    order_id=i.split('.')[0]
    #if int(order_id[-1])%2 != 1:
    #   continue
    #if 'scorpionOriginalData' not in json.load(open(file_path_name,encoding='utf-8'))['data']:
    #   continue
    count+=1
    #if '135638698918686720' not in i:
    #    continue
    #if count<3:
    #   continue
    #if count>3:
    #   break
    q.put(file_path_name)
    
print(count)
q.join()
now2=datetime.datetime.now() 
end=datetime.datetime.now()       
print('finish')
print("consume time %s:" %(now2-now1).seconds)
print('start_time:',start,'end_time:',end)
fwriter.close()




