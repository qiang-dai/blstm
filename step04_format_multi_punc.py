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
import pyIO
import tools

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
        add_cnt_dict(cleaned_punc_dict, word)
        #cleaned_punc_dict[word] = True
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

# 以字符串的形式读入所有数据
# def read_content(filename):
#     print(os.getcwd())
#     with open(filename, 'rb') as inp:
#         texts = inp.read().decode('utf8')
#     sentences = texts.split('\n')  # 根据换行切分
#     sentences = sentences[:threshold_line_cnt]
#     print(sentences[:300])
#     return sentences

def add_cnt_dict(cnt_dict, word):
    if word in cnt_dict:
        cnt_dict[word] += 1
    else:
        cnt_dict[word] = 1

def format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict):
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))

    total_res_list = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        res_list = clean_sentence(sentence)
        for res in res_list:
            add_cnt_dict(cnt_dict, res)
        #print(i, res_list)
        content_list = ['Header',]
        labels_list = [punc_list[0],]
        for w in res_list:
            ###如果是符号,就往前加
            if punctuation.is_punc(w[0]):
                ###统计词频
                add_cnt_dict(cleaned_punc_dict, w)

                ###如果符号连在一起, 就选择第一个符号
                if w not in punc_set:
                    #print('error: more than 1 flag:', w, sentence)
                    ###取最后的符号,忽略后面的符号
                    if w.find('...') != -1:
                        w = '...'
                    else:
                        w = w[-1]
                    if w not in punc_set:
                        ###实在奇怪的符号,就认为是空格
                        w = punc_list[0]
                        print('error: tail not punc:', w, sentence)
                labels_list[-1] = w
                #if w == '/':
                #    w = 'LEFT'
                ###可能多个符号在一起
                w = w.replace('/', 'LEFT')
            else:
                content_list.append(w)
                labels_list.append(punc_list[0])
        ###添加结束标记
        content_list.append('Tail')
        labels_list.append(punc_list[0])

        out_list = []
        for pos in range(len(content_list)):
            punc = labels_list[pos]
            if punc == punc_list[0]:
                pass
            else:
                pass
            text = content_list[pos] + '/' + punc
            out_list.append(text)
        total_res_list.append(' '.join(out_list))
    return total_res_list

if __name__ == '__main__':

    ###初始化:标点符号写文件
    punctuation.save_punc_list(punctuation.punctuation_list)

    ###<programe> raw_data/en_punctuation_recommend_train_100W  1000000 raw_data/res.txt
    filename, threshold_line_cnt, result_name = tools.args()

    ###词频统计
    cnt_dict = {
        'Unknown':1,
        'Header':1,
        'Tail':1,
    }

    ###标点符号频率统计
    cleaned_punc_dict = {}

    ###所有标点符号
    punc_list = punctuation.get_punc_list()

    ###两遍过滤
    ###1,遍历所有,判断标点符号的全集
    ###2,根据标点符号全集,生成训练数据
    sentences = pyIO.get_content(filename)
    res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict)
    print('res_list[:3]:', res_list[:3])

    ###统计平均值,然后过滤
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))
    print('res_list size:', len(res_list))
    total_cnt = 0
    for k in cleaned_punc_dict.keys():
        print(cleaned_punc_dict[k], '    ', k)
        total_cnt += cleaned_punc_dict[k]

    avg_cnt = total_cnt/len(cleaned_punc_dict) *2/3

    print('filtered...less than ', avg_cnt)
    for k in cleaned_punc_dict.keys():
        if cleaned_punc_dict[k] < avg_cnt and len(k) > 1:
            print('ignore', cleaned_punc_dict[k], '    ', k)

    print('valid...large than ', avg_cnt)
    cnt = 0
    for k in cleaned_punc_dict.keys():
        if cleaned_punc_dict[k] > avg_cnt or len(k) == 1:
            cnt += 1
            print(cnt, 'ignore', cleaned_punc_dict[k], '    ', k)
            if k not in punc_set:
                punc_list.append(k)

    ###再次计算
    print ('cnt_dict size:', len(cnt_dict))
    ###词频统计
    cnt_dict = {
        'Unknown':1,
        'Header':1,
        'Tail':1,
    }
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))
    ###存储punc列表

    res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict)
    print('res_list[:3]:', res_list[:3])
    print('res_list size:', len(res_list))
    #print(' '.join(res_list))

    ###保存标点符号
    punctuation.save_punc_list(punc_list)

    ###保存最终处理结果,格式是:
    ### Header/UNKNOWN by/UNKNOWN robert/UNKNOWN browning/. Tail/UNKNOWN
    pyIO.save_to_file('\n'.join(res_list), result_name)
    c = Counter(cnt_dict)
    print('add_cnt_dict:', len(cnt_dict), c.most_common(100))
    print ('cnt_dict size:', len(cnt_dict))
    
    for i, punc in enumerate(punc_list):
        print(i+1, punc)

