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

    print('sentence:', sentence)
    print('text:', text)
    tmp_dict = json.loads(text)
    
    extra_dict = tmp_dict['extra']
    if 'trainItem' in extra_dict:
        sessionId = extra_dict['trainItem']['sessionId']
        extra_dict = extra_dict['trainItem']['extra']
    else:
        print(extra_dict)
        sessionId = extra_dict['sessionId']

    item_dict = {}
    if sessionId in result_dict:
        item_dict = result_dict[sessionId]

    if 'responseTime' in extra_dict:
        item_dict['responseTime'] = extra_dict['responseTime']
    if 'reportTime' in extra_dict:
        item_dict['reportTime'] = extra_dict['reportTime']

    result_dict[sessionId] = item_dict

for k in result_dict.keys():
    #print(k, result_dict[k])
    ###判断时间差
    item_dict = result_dict[k]
    if len(item_dict) == 1 and 'responseTime' in item_dict:
        print (k, 'lost reportTime', item_dict)
    if len(item_dict) == 2:
        diff = int(item_dict['reportTime']) - int(item_dict['responseTime'])
        print (k, 'diff:', diff/1000, 's', item_dict)

