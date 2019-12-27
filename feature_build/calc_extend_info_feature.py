import gzip
import os
import json
import functools
from multiprocessing import Process, Queue
from feature_build.extend_info_feature_g2_l0 import RiskFeatureGroupExtendInfoG2

def fix_dic(extend_dict):
    out_dict = json.load(open('EXTEND_INFO_G2_FEATURE_PRO.json'))
    for key in out_dict:
        if key in extend_dict:
            out_dict[key] = extend_dict[key]

    return out_dict

# apply_time



process_num = 1
query_q = Queue()
result_q = Queue()
worker_p_list = []

filepath = 'D:/develop/test/model_build/feature_build/file'
# filepath = '/restore/working/tianjian/data/jiaka/report_jiaka/'
files = os.listdir(filepath)

re_count = 0

# mx_feature
def calc_feature(data, id):
    loan_no = data.strip('.txt.gz')
    json_data = {}
    try:
        print(loan_no)
        file_content = open(filepath + '/' + data, 'rb').read()
        data = gzip.decompress(file_content).decode('utf-8')
        json_data = json.loads(data)
    except OSError:
        try:
            file_content = open(filepath + '/' + data, 'rb').read()
            json_data = json.loads(file_content)
        except:
            pass



    try:
        data = json_data.get('data')
        if data:
            json_data = data
        m = RiskFeatureGroupExtendInfoG2()
        feat = m.calc_the_group(json_data)
        feat = fix_dic(feat)

    except Exception as e:
        raise Exception
        feat = fix_dic({})
        pass
    feat['order_id'] = loan_no

    global re_count
    re_count += 1
    print(loan_no,len(feat),re_count)
    if len(feat) != 711:
        print('长度不一致')
        return fix_dic({})


    return feat

def query_process(query, result_q):
    data, id = query
    feature = calc_feature(data, id)
    if feature is not None:
        result_q.put(feature)


def result_process_start():
    global sample_file
    pass
    sample_file = open('extend_info_feature_pro.csv', 'w', encoding='utf-8')
    # sample_file = open('test_one_extend_info_feature.csv', 'w', encoding='utf-8')


def result_process_end():
    sample_file.close()


def result_process(feature):
    title_value_list = []
    for key, value in feature.items():
        title_value_list.append((key, value))
    title_value_list = sorted(title_value_list, key=lambda x: x[0])
    title_list = list(map(lambda x: x[0], title_value_list))
    value_list = list(map(lambda x: x[1], title_value_list))
    #print(title_list)
    #print(value_list)
    return title_list, value_list


def query_worker(process_idx, query_q, result_q):
    while True:
        query = query_q.get()
        if query is None:
            result_q.put(None)
            return
        query_process(query, result_q)

def result_worker(result_q):
    result_process_start()
    none_cnt = 0
    line_cnt = 0
    first_title_list = None
    while True:
        result = result_q.get()
        if result is None:
            none_cnt += 1
            if none_cnt == process_num:
                result_process_end()
                return
            continue
        title_list, value_list = result_process(result)
        print(len(title_list),len(value_list))
        if line_cnt == 0:
            first_title_list = title_list
            sample_file.write(','.join(list(map(lambda x: str(x), first_title_list))) + '\n')

        else:
            if title_list != first_title_list:
                print('title_list does not match')
                print('first_title_list', )
                # print('title_list, ', title_list)
                print(set(title_list) - set(first_title_list))
                print(set(first_title_list) - set(title_list))

        temp = ','.join(list(map(lambda x: str(x), value_list))) + '\n'
        # print(temp)
        # if len(temp.split(',')) == 5751:
            # print(temp.split(','))
        sample_file.write(temp)
        # else:
            # print(temp.split(','))
        line_cnt += 1


if __name__ == '__main__':
    for process_idx in range(process_num):
        worker_p = Process(target=query_worker, args=(process_idx, query_q, result_q))
        worker_p.start()
        worker_p_list.append(worker_p)

    result_p = Process(target=result_worker, args=(result_q,))
    result_p.start()

    n = 0
    for file in files:
        n += 1
        # print(n)
        if n % 10000 == 0:
            print(n)
        # if n == test:
        #     break
        query_q.put((file, str(n)))

    for i in range(process_num):
        query_q.put(None)

    for p in worker_p_list:
        p.join()

    result_p.join()
print('finish')
