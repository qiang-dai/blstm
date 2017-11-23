# import numpy as np
# import re
# import time
import os,sys
# import punctuation
# import codecs
import json

filename = 'm.txt'
if len(sys.argv) > 1:
    filename = sys.argv[1]

# 以字符串的形式读入所有数据
print (os.getcwd())
with open(filename, 'rb') as inp:
    texts = inp.read().decode('utf8')
sentences = texts.split('\n')  # 根据换行切分
print (sentences[:10])

result_dict = {}
for sentence in sentences:
    pos = sentence.find('{')
    text = sentence[pos:]
    if len(sentence.strip()) == 0:
        continue

    if sentence.find('reportTime') != -1:
        i = 0

    #print('sentence:', sentence)
    #print('text:', text)
    try:
        tmp_dict = json.loads(text)
    except:
        pass
        continue
    
    extra_dict = tmp_dict['extra']

    ###特殊情况
    if not extra_dict:
        continue

    if 'trainItem' in extra_dict:
        sessionId = extra_dict['trainItem']['sessionId']
        extra_dict = extra_dict['trainItem']['extra']

        event_ts = extra_dict['event_ts'] + '000'
        kbTime = extra_dict['kbTime'] + '000'
        reportTime = extra_dict['reportTime']


        item_dict = {}
        item_dict['kbTime'] = kbTime
        item_dict['event_ts'] = event_ts
        item_dict['reportTime'] = reportTime
        result_dict[sessionId] = item_dict


cnt1_dict = {}
cnt2_dict = {}

def add_dict(some_dict, k, v):
    if k in some_dict:
        some_dict[k] += v
    else:
        some_dict[k] = v

for k in result_dict.keys():
    #print(k, result_dict[k])
    ###判断时间差
    item_dict = result_dict[k]
    diff1 = int(item_dict['event_ts']) - int(item_dict['kbTime'])
    diff2 = int(item_dict['reportTime']) - int(item_dict['event_ts'])
    #print (k, 'diff1:', diff1/1000, 'diff2:', diff2/1000, result_dict)
    print (k, 'diff1:', diff1/1000, 'diff2:', diff2/1000)

    k1 = int(diff1/1000)
    k1 = str(k1)
    k1 = len(k1)
    k1 = 10**k1
    add_dict(cnt1_dict, k1, 1)

    k2 = int(diff2/1000)
    k2 = str(k2)
    k2 = len(k2)
    k2 = 10**k2
    add_dict(cnt2_dict, k2, 1)

print ('cnt1_dict')
for k in cnt1_dict.keys():
    print (k, ':',cnt1_dict[k])

print ('cnt2_dict')
for k in cnt2_dict.keys():
    print (k,':', cnt2_dict[k])



