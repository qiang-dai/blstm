#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString

filename = sys.argv[1]
f = open(filename, 'r')
s = f.read()
t_list = s.split('\n')

###解析 monitor.log
res_list = []
for line in t_list:
    ###解析json
    timestamp = line.split('INFO')[0]
    pos = line.find('{')
    text = line[pos:]
    if len(line.strip()) == 0:
        continue
    tmp_dict = json.loads(text)

    ###过滤train
    if tmp_dict['operation'] != 'train':
        continue
    duid = tmp_dict['duid']

    ###屏蔽部分空训练
    if 'beforeMeanVarianceList' not in tmp_dict['paramExtra']:
        continue
    ###添加结果
    res = (tmp_dict['duid'], timestamp, tmp_dict['paramExtra']['beforeMeanVarianceList'], tmp_dict['paramExtra']['afterMeanVarianceList'], tmp_dict['paramExtra']['stickerIdList'], tmp_dict['flowExtra']['clsName'])
    res_list.append(res)

res_list.sort()

last_duid = ''
duid_pos_mean_dict = {}
for i,r in enumerate(res_list):
    r = res_list[i]
    ###parse duid,pos
    duid = r[0]
    pos = pyString.reExtractData('\[(\d+)\] vecotor', r[2][0], 1)
    mean = pyString.reExtractData('mean:(0.\d+)', r[2][0], 1)

    ###统计行数
    if duid not in duid_pos_mean_dict:
        duid_pos_mean_dict[duid] = {}
    key = pos + "_" + mean
    if key not in duid_pos_mean_dict:
        duid_pos_mean_dict[key] = 1
    else:
        duid_pos_mean_dict[key] += 1

    ###打印
    if last_duid != duid:
        last_duid = duid
        ###统计
        for key in duid_pos_mean_dict[last_duid]:
            val = duid_pos_mean_dict[last_duid][key]
            if val > 2:
                print ('warning', last_duid, key)

        print('\n\n')
    print (r)

    

