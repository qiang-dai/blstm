import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os
from collections import Counter
#from punctuation import punctuation_dict
import punctuation
import codecs

# 以字符串的形式读入所有数据
print (os.getcwd())
with open('raw_data/en_punctuation_recommend_train_100W', 'rb') as inp:
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
    if uchar in punctuation.punctuation_dict:
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

    ###这里判断一下,单词后面是
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
    'Unknown':1,
    'Header':1
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
    #if i < 10000:
    if True:
        #print(i, res_list)
        content_list = ['Header',]
        labels_list = [punctuation.punctuation_unkown,]
        for w in res_list:
            ###如果是符号,就往前加
            if is_punc(w[0]):
                labels_list[-1] = w
            else:
                content_list.append(w)
                labels_list.append(punctuation.punctuation_unkown)

        out_list = []
        for pos in range(len(content_list)):
            punc = labels_list[pos]
            if punc == punctuation.punctuation_unkown:
                pass
            else:
                #punc = punctuation.punctuation_dict[punc]
                pass
            text = content_list[pos] + '/' + punc
            out_list.append(text)
            
        print (out_list)
        write_file(filename, '  '.join(out_list))
        #for pos in range(len(content_list)):
        #    print
        #print (content_list)
        #print (labels_list)
    else:
        break

c = Counter(cnt_dict)
print ('add_cnt_dict:', len(cnt_dict), c.most_common(100))
print ('punc_dict:', punc_dict)
    ###写文件
    #write_file(filename, ' '.join([e for e in res_list]))

