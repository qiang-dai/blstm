#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString
import parse_train
import pyIO

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

    def my_key1(x):
        return x[0]+ x[2] + x[3] + x[1]
    res_list.sort()
    return res_list

def check_predict_error(res_list):
    predict_error_duid_dict = {}
    predict_error_session_dict = {}

    last_duid = ''
    for i,r in enumerate(res_list):
        if i >= len(res_list) - 1:
            break
        duid = r[0]

        r0 = res_list[i]
        r1 = res_list[i+1]
        if r0[0] == r1[0] \
            and r0[2] == r1[2] \
            and r0[3] == r1[3]:
            ###判断时间
            if r0[1][:18] != r1[1][:18]:
                print ('r0: error', r0)
                print ('r1: error', r1)

                key = r0[0]
                predict_error_duid_dict[key] = ''
                predict_error_session_dict[r[0] + '_' + r[-1]] = ''

                ###美观
                if last_duid != duid:
                    print ('\n')
                    last_duid = duid
    return predict_error_duid_dict, predict_error_session_dict
    #print (r)

if __name__ == '__main__':
    filename = sys.argv[1]
    t_list = pyIO.get_content(filename)

    train_res_list = parse_train.get_train_result(filename)
    predict_res_list = get_predict_result(t_list)

    ###思路：按时间顺序，排列所有的predict/train
    ### 然后分情况讨论
    ###1，丢失的train（2次predict之间，才可能丢失）
    ###2，延迟的train（2次predict之间，才可能延迟）
    ###3，正常的train，但是train错了（2次predict之间，才可能train错）

    ###1，排序所有的记录
    ###2，判断 2次predict之间


    ###预测错误的情况有几种
    ###1，丢失了train
    ###2，train延迟了
    ###3，train的不对（redis写入失败等情况）
    predict_error_duid_dict, predict_error_session_dict = check_predict_error(predict_res_list)
    print ('len(predict_error_duid_dict):', len(predict_error_duid_dict))
    print ('len(predict_error_session_dict)', len(predict_error_session_dict))

    ###训练数据
    train_item_dict = parse_train.get_train_key_by_file(filename)

    cnt = 0;
    duid_list = [r[0] for r in predict_res_list]
    duid_list = list(set(duid_list))

    ###丢失的train
    for k in predict_error_session_dict:
        if k not in train_item_dict:
            print ('lost train:', k)
            cnt += 1
        else:
            ###没有丢失/非延迟的train?不好区分
            print ('train error:', k)
    print("all predict cnt:", len(predict_res_list) - len(duid_list))
    print('total lost train cnt:', cnt)
    #print (train_item_dict)





