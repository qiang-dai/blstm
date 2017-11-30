#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString
import pyIO

###解析 monitor.log
def get_train_result(t_list):
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
        if tmp_dict['operation'] != 'train':
            continue
        duid = tmp_dict['duid']

        ###屏蔽部分空训练
        if 'beforeMeanVarianceList' not in tmp_dict['paramExtra']:
            continue
        ###添加结果
        res = (tmp_dict['duid'], timestamp, tmp_dict['paramExtra']['beforeMeanVarianceList'], tmp_dict['paramExtra']['afterMeanVarianceList'], tmp_dict['paramExtra']['stickerIdList'], tmp_dict['flowExtra']['clsName'], tmp_dict['sessionId'])
        res_list.append(res)

    res_list.sort()
    return res_list;

def get_train_cnt(res_list):
    last_duid = ''
    duid_pos_mean_dict = {
        last_duid:{}
    }
    for i,r in enumerate(res_list):
        r = res_list[i]
        ###parse duid,pos
        duid = r[0]
        pos = pyString.reExtractData('\[(\d+)\] vecotor', r[2][0], 1)
        mean = pyString.reExtractData('mean:([-]?\d+.\d+)', r[2][0], 1)

        ###统计行数
        if duid not in duid_pos_mean_dict:
            duid_pos_mean_dict[duid] = {}
        key = pos + "_" + mean
        if key not in duid_pos_mean_dict[duid]:
            duid_pos_mean_dict[duid][key] = 1
        else:
            duid_pos_mean_dict[duid][key] += 1

        ###打印
        if last_duid != duid:
            ###change
            last_duid = duid

            print('\n\n')
        print (r)
    return duid_pos_mean_dict


def parse_error(duid_pos_mean_dict):
    ###统计
    for duid in duid_pos_mean_dict:
        for key in duid_pos_mean_dict[duid]:
            val = duid_pos_mean_dict[duid][key]
            if val == 1:
                print ('warning', duid, key, val)
            elif val > 2:
                print ('error', duid, key, val)
            else:
                print ('info_ok', duid, key, val)

def get_train_key_by_file(filename):
    t_list = pyIO.get_content(filename)
    res_list = get_train_result(t_list)
    train_item_dict = get_train_key(res_list)
    #print(train_item_dict)
    return train_item_dict

def get_train_key(res_list):
    train_item_dict = {}
    for r in res_list:
        key = r[0] + '_' + r[-1]
        train_item_dict[key] = ''
    return train_item_dict

if __name__ == '__main__':
    filename = sys.argv[1]
    f = open(filename, 'r')
    s = f.read()
    t_list = s.split('\n')
    res_list = get_train_result(t_list)
    duid_pos_mean_dict = get_train_cnt(res_list)
    parse_error(duid_pos_mean_dict)

    train_item_dict = get_train_key_by_file(filename)
    print(train_item_dict)