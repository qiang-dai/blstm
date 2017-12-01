import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os,sys
from collections import Counter
import punctuation
import codecs

cleaned_punc_dict = {}
punc_set = set(punctuation.punctuation_list)

threshold_line_cnt = 1000000
if len(sys.argv) > 1:
    threshold_line_cnt = int(sys.argv[1])

# 以字符串的形式读入所有数据
print (os.getcwd())
with open('raw_data/en_punctuation_recommend_train_100W', 'rb') as inp:
    texts = inp.read().decode('utf8')
sentences = texts.split('\n')  # 根据换行切分
sentences = sentences[:threshold_line_cnt]
print (sentences[:300])

def getCharType(c):
    if punctuation.is_alphabet(c):
        return 1
    if punctuation.is_number(c):
        return 2
    if punctuation.is_punc(c):
        return 3
    return 4

###都使用小写字母
def transform(word):
    ###如果都是数字,就改为Num3 这种
    if punctuation.is_number(word[0]):
        word = 'num%d'%(len(word))
    elif punctuation.is_alphabet(word[0]):
        word = word.lower()
    elif punctuation.is_punc(word[0]):
        ###debug
        cleaned_punc_dict[word] = True
    else:
        pass
    return word.lower()

def merge_chars(word):
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
    ###查错
    res_list.append(transform(res))

    return res_list

def clean_sentence(sentence):
    word_list = sentence.split(" ")
    res_list = []
    for word in word_list:
        word = word.strip()
        if len(word) == 0:
            continue
        tmp_list = merge_chars(word)
        res_list.extend(tmp_list)

    return res_list

def clear_file(filename):
    f = codecs.open(filename, 'w', 'utf8')
    f.close()

def write_file(filename, res):
    f = codecs.open(filename, 'a+', 'utf8')
    f.write(res + '\n')
    f.close()

filename = 'raw_data/res.txt'
clear_file(filename)

cnt_dict = {
    'SP':1,
    'Header':1,
    'Tail':1,
}

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
    #print(i, res_list)
    content_list = ['Header',]
    labels_list = [punctuation.punctuation_list[0],]
    for w in res_list:
        ###如果是符号,就往前加
        if punctuation.is_punc(w[0]):
            ###如果符号连在一起, 就选择第一个符号
            if w not in punc_set:
                print('error: more than 1 flag:', w, sentence)
                ###取最后的符号,忽略后面的符号
                if w.find('...') != -1:
                    w = '...'
                else:
                    w = w[-1]
                if w not in punc_set:
                    ###实在奇怪的符号,就认为是空格
                    w = punctuation.punctuation_list[0]
                    print('error: tail not punc:', w, sentence)
            labels_list[-1] = w
            if w == '/':
                w = 'LEFT'
        else:
            content_list.append(w)
            labels_list.append(punctuation.punctuation_list[0])
    ###添加结束标记
    content_list.append('Tail')
    labels_list.append(punctuation.punctuation_list[0])

    out_list = []
    for pos in range(len(content_list)):
        punc = labels_list[pos]
        if punc == punctuation.punctuation_list[0]:
            pass
        else:
            pass
        text = content_list[pos] + '/' + punc
        out_list.append(text)

    #print (out_list)
    write_file(filename, '  '.join(out_list))

c = Counter(cnt_dict)
print ('add_cnt_dict:', len(cnt_dict), c.most_common(100))
for k in cleaned_punc_dict.keys():
    print (k)

