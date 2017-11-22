import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os
from collections import Counter
from punctuation import punctuation_dict

# 以字符串的形式读入所有数据
print (os.getcwd())
with open('/Users/xinmei365/Downloads/en_punctuation_recommend_train_100W', 'rb') as inp:
    texts = inp.read().decode('utf8')
sentences = texts.split('\n')  # 根据换行切分
print (sentences[:300])

# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if u'\u4e00' <= uchar<=u'\u9fff':
        return True
    else:
        return False

# 判断一个unicode是否是数字
def is_number(uchar):
    if u'\u0030' <= uchar and uchar<=u'\u0039':
        return True
    else:
        return False

# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False
def is_punc(uchar):
    if uchar in punctuation_dict:
        return True
    return False

def getCharType(c):
    if is_alphabet(c):
        return 1
    if is_number(c):
        return 2
    if is_punc(c):
        return 3
    return 4

punc_dict = {}
def add_punc(word):
    punc_dict[word] = True

###都使用小写字母
def transform(word):
    ###如果都是数字,就改为Num3 这种
    if is_number(word[0]):
        return 'num%d'%(len(word))
    elif is_alphabet(word[0]):
        pass
    elif is_punc(word[0]):
        add_punc(word)
    else:
        pass
    return word.lower()


def clean_word(word):
    #字符相连,数字相连,符号
    flag_list = [getCharType(c) for c in word]
    ###合并同类项
    last = flag_list[0]
    res = ''
    res_list = []
    for i in range(len(flag_list)):
        cur = flag_list[i]
        ###合并
        if cur == last:
            res += word[i]
        ###另外的单词
        else:
            res_list.append(transform(res))
            res = word[i]

        last = cur
    res_list.append(transform(res))

    return res_list

def clean_sentence(sentence):
    word_list = sentence.split(" ")
    res_list = []
    for word in word_list:
        word = word.strip()
        if len(word) == 0:
            continue
        tmp_list = clean_word(word)
        res_list.extend(tmp_list)

    return res_list

# def clear_file(filename):
#     f = open(filename, 'w')
#     f.close()
#
# def write_file(filename, res):
#     f = open(filename, 'w+')
#     f.write(res)
#     f.close()
#
# filename = 'res.txt'
# clear_file(filename)

cnt_dict = {}
def add_cnt_dict(cnt_dict, word):
    if word in cnt_dict:
        cnt_dict[word] += 1
    else:
        cnt_dict[word] = 1

for i in range(len(sentences)):
    sentence = sentences[i]
    res_list = clean_sentence(sentence)
    for res in res_list:
        add_cnt_dict(cnt_dict, res)
    if i < 10000:
        print(i, res_list)
    else:
        break

c = Counter(cnt_dict)
print ('add_cnt_dict:', c.most_common(1000))
print ('punc_dict:', punc_dict)
    ###写文件
    #write_file(filename, ' '.join([e for e in res_list]))