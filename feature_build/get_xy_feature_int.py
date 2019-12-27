import json
n=0
title=['order_id',
'behavior_score',
'behavior_believe',
'loan_order_cnt',
'loan_ended_order_cnt',
'overdue_order_cnt',
'loan_organ_cnt',
'consume_organ_cnt',
'net_organ_cnt',
'loan_cnt_1m',
'loan_cnt_3m',
'loan_cnt_6m',
'his_payback_success_cnt',
'his_payback_fail_cnt',
'his_payback_success_cnt_1m',
'his_payback_fail_cnt_1m',
'loan_ages',
'days_to_last_loan_date',
'loan_cnt_1m_vs_3m',
'loan_cnt_1m_vs_6m',
'his_payback_success_cnt_1m_vs_all',
'his_payback_fail_cnt_1m_vs_all',
'loan_order_cnt_avg_day',
'loan_organ_cnt_avg_day',
'loan_order_cnt_avg_org',
'loan_ended_order_cnt_rate',
'his_payback_success_rate',
'his_payback_success_rate_1m',
'overdue_order_cnt_rate',
'consume_organ_cnt_rate']
file=open(r'/restore/working/tianjian/calc_xinyan_feature/data/jk_0608_xy_feautre_int.txt','w',encoding='utf-8')
file.write(','.join(title)+'\n')
count=0
for line in open(r'/restore/working/tianjian/calc_xinyan_feature/data/20190826113215tmp_0608_jk_xy_feature_int.csv',encoding='utf-8'):
    feature_list=[]
    # print(line)
    count+=1
    if count==1:
        continue
    # if count>5:
    #     break
    # if n>19:
    #     break
    data=line.strip().split('\t')
    feature = data[0].replace('\\\\"', '\\"')
    feature = feature.replace("\"\"", '"')
    feature = feature.replace("}\"", '}')
    feature = feature.replace("\"{", '{')
    # print(feature)
    tmp = feature.split('},')
    feature = tmp[0]+'}'
    order_id = tmp[1]
    feature=json.loads(feature)
    print(count)
    behavior_score = feature['behavior_score']
    behavior_believe = feature['behavior_believe']
    loan_order_cnt = feature['loan_order_cnt']
    loan_ended_order_cnt = feature['loan_ended_order_cnt']
    overdue_order_cnt = feature['overdue_order_cnt']
    loan_organ_cnt = feature['loan_organ_cnt']
    consume_organ_cnt = feature['consume_organ_cnt']
    net_organ_cnt = feature['net_organ_cnt']
    loan_cnt_1m = feature['loan_cnt_1m']
    loan_cnt_3m = feature['loan_cnt_3m']
    loan_cnt_6m = feature['loan_cnt_6m']
    his_payback_success_cnt = feature['his_payback_success_cnt']
    his_payback_fail_cnt = feature['his_payback_fail_cnt']
    his_payback_success_cnt_1m = feature['his_payback_success_cnt_1m']
    his_payback_fail_cnt_1m = feature['his_payback_fail_cnt_1m']
    loan_ages = feature['loan_ages']
    try:
        days_to_last_loan_date = feature['days_to_last_loan_date']
    except:
        days_to_last_loan_date = 'NaN'
    try:
        loan_cnt_1m_vs_3m = feature['loan_cnt_1m_vs_3m']
        loan_cnt_1m_vs_6m = feature['loan_cnt_1m_vs_6m']
    except:
        loan_cnt_1m_vs_3m = feature['loan_cnt_1m'] / feature['loan_cnt_3m'] if feature['loan_cnt_3m'] != 0 and feature[
                                                                                                                   'loan_cnt_3m'] != 'NaN' else 'NaN'
        loan_cnt_1m_vs_6m = feature['loan_cnt_1m'] / feature['loan_cnt_6m'] if feature['loan_cnt_6m'] != 0 and feature[
                                                                                                                   'loan_cnt_6m'] != 'NaN' else 'NaN'

    try:
        his_payback_success_cnt_1m_vs_all = feature['his_payback_success_cnt_1m_vs_all']
        his_payback_fail_cnt_1m_vs_all = feature['his_payback_fail_cnt_1m_vs_all']
    except:
        his_payback_success_cnt_1m_vs_all = feature['his_payback_success_cnt_1m'] / feature[
            'his_payback_success_cnt'] if feature['his_payback_success_cnt'] != 0 and feature[
                                                                                          'his_payback_success_cnt'] != 'NaN' else 'NaN'
        his_payback_fail_cnt_1m_vs_all = feature['his_payback_fail_cnt_1m'] / feature['his_payback_fail_cnt'] if \
        feature['his_payback_fail_cnt'] != 0 and feature['his_payback_fail_cnt'] != 'NaN' else 'NaN'

    try:
        loan_order_cnt_avg_day = feature['loan_order_cnt_avg_day']
    except:
        loan_order_cnt_avg_day = feature['loan_order_cnt'] / feature['loan_ages'] if feature['loan_ages'] != 0 and \
                                                                                     feature[
                                                                                         'loan_ages'] != 'NaN' else 'NaN'

    try:
        loan_organ_cnt_avg_day = feature['loan_organ_cnt_avg_day']
    except:
        loan_organ_cnt_avg_day = feature['loan_organ_cnt'] / feature['loan_ages'] if feature['loan_ages'] != 0 and \
                                                                                     feature[
                                                                                         'loan_ages'] != 'NaN' else 'NaN'
    try:
        loan_order_cnt_avg_org = feature['loan_order_cnt_avg_org']
    except:
        loan_order_cnt_avg_org = feature['loan_order_cnt'] / feature['loan_organ_cnt'] if feature[
                                                                                              'loan_organ_cnt'] != 0 and \
                                                                                          feature[
                                                                                              'loan_organ_cnt'] != 'NaN' else 'NaN'

    try:
        loan_ended_order_cnt_rate = feature['loan_ended_order_cnt_rate']
    except:
        loan_ended_order_cnt_rate = feature['loan_ended_order_cnt'] / feature['loan_order_cnt'] if feature[
                                                                                                       'loan_order_cnt'] != 'NaN' or \
                                                                                                   feature[
                                                                                                       'loan_order_cnt'] != 0 else 'NaN'

    try:
        his_payback_success_rate = feature['his_payback_success_rate']
    except:
        his_payback_success_rate = feature['his_payback_success_cnt'] / (
        feature['his_payback_success_cnt'] + feature['his_payback_fail_cnt']) if feature[
                                                                                     'his_payback_fail_cnt'] != 'NaN' and int(
            feature['his_payback_success_cnt']) != 'NaN' and int(
            (feature['his_payback_success_cnt'] + feature['his_payback_fail_cnt'])) != 0 else 'NaN'

    try:
        his_payback_success_rate_1m = feature['his_payback_success_rate_1m']
    except:
        his_payback_success_rate_1m = feature['his_payback_success_cnt_1m'] / (
        feature['his_payback_success_cnt_1m'] + feature['his_payback_fail_cnt_1m']) if int(
            feature['his_payback_fail_cnt_1m']) != 'NaN' and int(
            feature['his_payback_success_cnt_1m'] + feature['his_payback_fail_cnt_1m']) != 0 else 'NaN'

    try:
        overdue_order_cnt_rate = feature['overdue_order_cnt_rate']
    except:
        overdue_order_cnt_rate = feature['overdue_order_cnt'] / feature['loan_order_cnt'] if int(
            feature['loan_order_cnt']) != 0 and int(feature['loan_order_cnt']) != 'NaN' else 'NaN'

    try:
        consume_organ_cnt_rate = feature['consume_organ_cnt_rate']
    except:
        consume_organ_cnt_rate = feature['consume_organ_cnt'] / feature['loan_organ_cnt'] if feature[
                                                                                                 'loan_organ_cnt'] != 'NaN' and \
                                                                                             feature[
                                                                                                 'loan_organ_cnt'] != 0 else 'NaN'

    feature_list.append(order_id)
    feature_list.append(behavior_score)
    feature_list.append(behavior_believe)
    feature_list.append(loan_order_cnt)
    feature_list.append(loan_ended_order_cnt)
    feature_list.append(overdue_order_cnt)
    feature_list.append(loan_organ_cnt)
    feature_list.append(consume_organ_cnt)
    feature_list.append(net_organ_cnt)
    feature_list.append(loan_cnt_1m)
    feature_list.append(loan_cnt_3m)
    feature_list.append(loan_cnt_6m)
    feature_list.append(his_payback_success_cnt)
    feature_list.append(his_payback_fail_cnt)
    feature_list.append(his_payback_success_cnt_1m)
    feature_list.append(his_payback_fail_cnt_1m)
    feature_list.append(loan_ages)
    feature_list.append(days_to_last_loan_date)
    feature_list.append(loan_cnt_1m_vs_3m)
    feature_list.append(loan_cnt_1m_vs_6m)
    feature_list.append(his_payback_success_cnt_1m_vs_all)
    feature_list.append(his_payback_fail_cnt_1m_vs_all)
    feature_list.append(loan_order_cnt_avg_day)
    feature_list.append(loan_organ_cnt_avg_day)
    feature_list.append(loan_order_cnt_avg_org)
    feature_list.append(loan_ended_order_cnt_rate)
    feature_list.append(his_payback_success_rate)
    feature_list.append(his_payback_success_rate_1m)
    feature_list.append(overdue_order_cnt_rate)
    feature_list.append(consume_organ_cnt_rate)
    file.write(','.join(map(lambda x:str(x),feature_list))+'\n')
file.close()


