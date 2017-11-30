#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString
import parse_train

###解析 monitor.log
def get_predict_result(t_list):
    res_list = []
    for line in t_list:
        ###解析json
        timestamp = line.split('INFO')[0]
        pos = line.find('{')
        text = line[pos:]
        if len(line.strip()) == 0:
            continue
        tmp_dict = json.loads(text)

        if 'operation' not in tmp_dict:
            continue
        ###过滤train
        if tmp_dict['operation'] != 'predict':
            continue
        duid = tmp_dict['duid']

        ###屏蔽部分空训练
        #if 'beforeMeanVarianceList' not in tmp_dict['paramExtra']:
        #    continue
        ###添加结果
        res = (tmp_dict['duid'], timestamp, tmp_dict['paramExtra']['itemId'], tmp_dict['category'], tmp_dict['paramExtra']['score'], tmp_dict['extra']['taghit'], tmp_dict['sessionId'])
        res_list.append(res)

    res_list.sort()
    return res_list

def check_predict_error(res_list):
    predict_error_dict = {}

    for i,r in enumerate(res_list):
        if i >= len(res_list) - 1:
            break

        r0 = res_list[i]
        r1 = res_list[i+1]
        if r0[0] == r1[0] \
            and r0[2] == r1[2] \
            and r0[3] == r1[3]:
            ###判断时间
            if r0[1][:18] != r1[1][:18]:
                print ('error', r0)
                print ('error', r1)
                print ('\n')

                key = r0[0] + '_' + r0[-1]
                predict_error_dict[key] = ''
    return predict_error_dict
    #print (r)

if __name__ == '__main__':
    filename = sys.argv[1]
    f = open(filename, 'r')
    s = f.read()
    t_list = s.split('\n')
    res_list = get_predict_result(t_list)
    predict_error_dict = check_predict_error(res_list)
    #print (predict_error_dict)

    ###训练数据
    train_item_dict = parse_train.get_train_key_by_file(filename)

    cnt = 0;
    duid_list = [r[0] for r in res_list]
    duid_list = list(set(duid_list))

    for k in predict_error_dict:
        if k not in train_item_dict:
            print ('lost train:', k)
            cnt += 1
    print("all predict cnt:", len(res_list) - len(duid_list))
    print('total lost train cnt:', cnt)
    #print (train_item_dict)






