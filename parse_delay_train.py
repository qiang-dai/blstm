#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString
import pyIO
import time


###解析 monitor.log
def get_total_result(t_list):
    res_list = []
    for line in t_list:
        ###解析json
        timestamp = line.split('INFO')[0]
        pos = line.find('{')
        text = line[pos:]
        if len(line.strip()) == 0:
            continue
        print('text:', text)
        tmp_dict = json.loads(text)

        ###添加结果
        if tmp_dict['operation'] == 'predict':
            res = (tmp_dict['duid'], timestamp, tmp_dict['sessionId'], tmp_dict['operation'], tmp_dict['paramExtra']['itemId'], tmp_dict['category'], tmp_dict['paramExtra']['score'], tmp_dict['timestamp'])
        else:
            res = (tmp_dict['duid'], timestamp, tmp_dict['sessionId'], tmp_dict['operation'])

        res_list.append(res)

    res_list.sort()
    ###显示长度
    print('len(t_list):', len(t_list))
    print('len(res_list):', len(res_list))
    return res_list

if __name__ == '__main__':
    filename = sys.argv[1]
    t_list = pyIO.get_content(filename)

    ###所有数据
    total_result = get_total_result(t_list)

    ###所有duid
    duid_list = list(set([e[0] for e in total_result]))
    print('len(duid_list):', len(duid_list))

    ###所有训练集合:session -> item
    total_train_dict = dict([(e[2], e) for e in total_result if e[3] == 'train'])
    # for duid in duid_list:
    #     tmp_list = [e for e in total_result if e[0] == duid]
    #     #print(tmp_list)

    ###所有预测集合
    total_predict_dict = {}
    total_predict_list = [e for e in total_result if e[3] == 'predict']
    for e in total_predict_list:
        key = e[0] + '_' + e[4] + '_' + e[5]
        if key not in total_predict_dict:
            total_predict_dict[key] = []
        total_predict_dict[key].append(e)

    ###排序
    def sort_predict(x):
        ###duid,item,tag,time
        return x[0] + x[4] + x[5] + x[1]
    for k in total_predict_dict.keys():
        v = total_predict_dict[k]
        v.sort(key = sort_predict)
        total_predict_dict[k] = v

    ###判断延迟的情况
    predict_check_cnt = 0
    train_lost_cnt = 0
    train_delay_cnt = 0
    train_between_cnt = 0

    diff_dict = {}
    for k,v in total_predict_dict.items():
        for i, e in enumerate(v):
            if i == 0:
                continue
            ###判断前一个
            predict_check_cnt += 1

            ###lost train
            if v[i-1][2] not in total_train_dict:
                train_lost_cnt += 1
                print('i-1, lost train:', v[i-1])
                print('i  , lost train:', v[i])
                timestamp = v[i-1][7]
                current_time = int(round(time.time() * 1000))
                diff = '%s'%( int((current_time - timestamp)/1800000) )
                if diff not in diff_dict:
                    diff_dict[diff] = 1
                else:
                    diff_dict[diff] += 1




            # score = e[6]
            # sessionId = e[2]
            ###如果score相同，那么就是重复发送
            if v[i-1][6] == v[i][6] and v[i-1][2] in total_train_dict:
                print('i-1, delay train:', v[i-1])
                print('i  , delay train:', v[i])
                ###验证训练时间是否在v[i]之后
                sessionId = v[i-1][2]
                t = total_train_dict[sessionId]
                if t[1] < v[i][1]:
                    train_between_cnt += 1
                    print('bad,between train:', t)
                    print('\n')
                else:
                    train_delay_cnt += 1
                    print('ok, check delay:', t)
                    print('\n')
        print ('-'*15)

    print ('predict_check_cnt',predict_check_cnt)
    print ('train_lost_cnt',train_lost_cnt, '%.2f%%'%(train_lost_cnt/predict_check_cnt*100))
    print ('train_delay_cnt',train_delay_cnt, '%.2f%%'%(train_delay_cnt/predict_check_cnt*100))
    print ('train_between_cnt',train_between_cnt, '%.2f%%'%(train_between_cnt/predict_check_cnt*100))
    print ('diff_dict:', diff_dict)






