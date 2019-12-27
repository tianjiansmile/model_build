# coding: utf-8
import pymysql
import datetime
import requests

#  获取数据库连接
def getPymysql(ahost,auser,pw,Db,aport):
    # read_con = pymysql.connect(host='rm-uf6j98p717px3v8pj5o.mysql.rds.aliyuncs.com', user='tianshen',
    #                            password='tianshen', database='tianshen_syn_temp', port=3306, charset='utf8')

    read_con = pymysql.connect(host=ahost, user=auser,
                               password=pw, database=Db, port=aport, charset='utf8')
    # 创建游标
    cursor = read_con.cursor()

    return read_con

apply_count = 0
avg_apply = 0

def get_timeback(loan_no,cursor):
    sql = "select uo.create_time,ub.identity_no_ency from user_order uo " \
          "left join user_basics_info ub on ub.user_id = uo.user_id where merchant_id='11' and loan_order_no = '%s'"

    cursor.execute(sql  % (loan_no))
    re = cursor.fetchone()
    # for re in result:
    print(re[0],re[1])
    url = 'http://172.19.133.11:8023/getJinpanLoanFeatures?identityNo=%s&create_time=%s'
    currDate = re[0]
    idNum = re[1]
    res=requests.get(url % (idNum,currDate))
    if res.status_code==200:
        all_list=[]
        res=res.json()
        result=res.get('result')
        fea = result.get('features')
        user_fea = fea.get('user_feature')
        apply_sum = user_fea.get('apply_sum_all')
        print(apply_sum)
        if apply_sum != 'NaN':
            global apply_count
            global avg_apply
            apply_count += 1
            avg_apply += int(apply_sum)



if __name__ == '__main__':
    import time
    time1 = time.time()


    # 读数据
    read_con = getPymysql('139.196.75.48', 'root', 'zhouao.123',
                          'user_db', 8066)
    cursor = read_con.cursor()

    with open('D:/特征提取/嘉卡/20190801095422tmp_jk.txt','r') as rf:
        for line in rf.readlines()[1000:1200]:
            line = line.split(',')
            loan_no = line[2]
            create_time = line[0]
            print(loan_no,create_time)

            get_timeback(loan_no,cursor)

    print(apply_count,avg_apply)

    cursor.close()
    read_con.close()

    time2 = time.time()
    print(time2 - time1)