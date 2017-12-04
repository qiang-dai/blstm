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
import datetime

threshold_word_cnt = 10

def getCharType(c):
    if punctuation.is_alphabet(c):
        return 1
    if punctuation.is_number(c):
        return 2
    if punctuation.is_punc(c):
        return 3
    if punctuation.is_emoji(c):
        return 4
    return 5

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
    ###新闻的处理
    sentence = sentence.replace('\n', ' ')
    ###首先根据空格\空白字符进行处理
    sentence = re.sub('\s',' ',sentence)
    word_list = sentence.split(" ")
    res_list = []
    for word in word_list:
        word = word.strip()
        if len(word) == 0:
            continue
        tmp_list = merge_chars(word)
        res_list.extend(tmp_list)

    return res_list

def add_cnt_dict(cnt_dict, word):
    if word in cnt_dict:
        cnt_dict[word] += 1
    else:
        cnt_dict[word] = 1

def transform_punc(w, punc_list, punc_set):
    w = w.replace('/', 'LEFT')
    if w not in punc_set:
        w = punc_list[0]
    return w

def format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict):
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))

    total_res_list = []
    for i in range(len(sentences)):
        if i%100 == 0:
            print('sentence total, i:',len(sentences), i)

        sentence = sentences[i]
        # sub_sentence_list = sentence.split('\\n')
        # for sub_sentence in sub_sentence_list:
        if True:
            sub_sentence = sentence.replace('\\n', ' ')
            ###解析结果
            res_list = clean_sentence(sub_sentence)
            for res in res_list:
                add_cnt_dict(cnt_dict, res)

            ###过滤无标点句子
            flag_punc_find = False

            content_list = ['Header',]
            labels_list = [punc_list[0],]
            for w in res_list:
                ###如果是符号,就往前加
                if punctuation.is_punc(w[0]):
                    flag_punc_find = True

                    if w not in punc_set:
                        w = punctuation.get_punc_other()

                    ###统计词频
                    add_cnt_dict(cleaned_punc_dict, w)
                    ###标点符号进行替换
                    w = transform_punc(w, punc_list, punc_set)
                    labels_list[-1] = w
                else:
                    if punctuation.is_emoji(w[0]):
                        w = 'EMOJI'

                    content_list.append(w)
                    labels_list.append(punc_list[0])

            ###添加结束标记
            content_list.append('Tail')
            labels_list.append(punc_list[0])

            ###如果一个标点都没有，就忽略这句话
            if not flag_punc_find:
                continue

            out_list = []
            for pos, word in enumerate(content_list):
                out_list.append([word, labels_list[pos]])
            total_res_list.append(out_list)
    return total_res_list

def get_min_punc_cnt(cleaned_punc_dict):
    min_cnt = 0
    for k in punctuation.punctuation_list:
        if k in cleaned_punc_dict and cleaned_punc_dict[k] < min_cnt:
            print('get_min:', k, cleaned_punc_dict[k])
            min_cnt = cleaned_punc_dict[k]

    print('min_cnt:', min_cnt)
    return min_cnt

def get_total_cnt(some_cnt_dict, limit):
    total_cnt_dict = {}
    for k in some_cnt_dict:
        v = some_cnt_dict[k]
        if v >= limit:
            total_cnt_dict[k] = True

    for k in some_cnt_dict:
        if k not in total_cnt_dict:
            print('lost word limit:', limit, k)
    return len(total_cnt_dict)

def get_args():
    filename = 'raw_data/total_english.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    threshold_line_cnt = 10000
    if len(sys.argv) > 2:
        threshold_line_cnt = int(sys.argv[2])

    res_file = 'raw_data/total_english.txt'
    if len(sys.argv) > 3:
        res_file = sys.argv[3]

    flag_ignore_complex_punc = False
    if len(sys.argv) > 4:
        if sys.argv[4] == 'True':
            flag_ignore_complex_punc = True

    return filename, threshold_line_cnt, res_file, flag_ignore_complex_punc

if __name__ == '__main__':

    ###初始化:标点符号写文件
    punctuation.save_punc_list(punctuation.punctuation_list)

    ###<programe> raw_data/en_punctuation_recommend_train_100W  1000000 raw_data/res.txt
    filename, threshold_line_cnt, result_name, flag_ignore_complex_punc = get_args()

    ###词频统计
    cnt_dict = {
        'SP':threshold_word_cnt,
        'Header':threshold_word_cnt,
        'Tail':threshold_word_cnt,
    }

    ###标点符号频率统计
    cleaned_punc_dict = {}

    ###所有标点符号
    punc_list = punctuation.get_punc_list()
    sentences = pyIO.get_content(filename)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'pyIO.get_content:', filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'sentences length:', len(sentences))

    if len(sentences) > threshold_line_cnt:
        sentences = sentences[:threshold_line_cnt]
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'sentences length:', len(sentences))

    punc_set = set(punc_list)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'punc_set size:', len(punc_set))

    ###词频统计
    cnt_dict = {
        'None' : threshold_word_cnt,
        'Header' : threshold_word_cnt,
        'Tail' : threshold_word_cnt,
    }
    punc_set = set(punc_list)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'punc_set size:', len(punc_set))
    ###存储punc列表
    res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'res_list[:3]:', res_list[:3])
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'res_list size:', len(res_list))

    ###保存标点符号
    punctuation.save_punc_list(punc_list)

    ###保存最终处理结果,格式是:
    content_line_list = []
    ###统计

    for i, tmp_list in enumerate(res_list):
        line_list = []
        for word,punc in tmp_list:
            if word in cnt_dict and cnt_dict[word] < threshold_word_cnt:
                ###填充字符
                word = punctuation.get_filled_word()
            if punc not in punc_set:
                print('[%d] error punc:'%i, punc, tmp_list)
                punc = punc_list[0]

            line_list.append('%s/%s'%(word, punc))
        content_line_list.append(' '.join(line_list))

    ### Header/SP by/SP robert/SP browning/. Tail/SP
    pyIO.save_to_file('\n'.join(content_line_list), result_name)

    c = Counter(cnt_dict)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'add_cnt_dict:', len(cnt_dict), c.most_common(100))
    for k in cnt_dict.keys():
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'word', cnt_dict[k], k)

    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'total word cnt:', get_total_cnt(cnt_dict, 0))
    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'total cnt >= 10 word cnt:', get_total_cnt(cnt_dict, threshold_word_cnt))
    # print ('total cnt >= 20 word cnt:', get_total_cnt(cnt_dict, 20))

    # ###图形显示长度
    # plt_val_cnt_dict = {}
    # for cnt in cnt_dict.keys():
    #     num = '%d'%(cnt_dict[cnt])
    #     add_cnt_dict(plt_val_cnt_dict, num)
    #
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1) # 创建图表1
    # for i in range(10000):
    #     key = '%d'%i
    #     val_cnt = 0
    #     #print ('key,val_cnt_dict:', key, plt_val_cnt_dict)
    #     if key in plt_val_cnt_dict:
    #         val_cnt = plt_val_cnt_dict[key]
    #         if (1 < val_cnt < 20 ):
    #             plt.plot(i, val_cnt, 'or')
    # plt.show()



