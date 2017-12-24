import time
import os,sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pyIO
import step04_format_multi_punc
import punctuation
import pickle
from tensorflow.contrib import rnn
import datetime

### 设置显存根据需求增长
import numpy as np
import tools
import datetime
import subprocess
import fasttext
import step05_append_category
import numpy as np
import punctuation

def append_cnt_dict(cnt_dict, k, v):
    if k not in cnt_dict:
        cnt_dict[k] = []
    cnt_dict[k].append(v)

def add_cnt_dict(cnt_dict, k, v):
    if k in cnt_dict:
        v += cnt_dict[k]
    cnt_dict[k] = v

def getRate(cnt_dict, size):
    rate_dict = {}

    for k,v in cnt_dict.items():
        print('%-10s'%k, v, v/size)
        rate_dict[k] = v/size
    return rate_dict

def get_file_rate_list(file_dir):
    file_dir = sys.argv[1]
    filename_list = tools.get_filename_list(file_dir)

    file_rate_list = []
    ###每个目录取1000行
    for index, filename in enumerate(filename_list):
        cnt_dict = {}

        total_list = tools.get_total_limit_list(filename, 2000*10000)
        for line in total_list:
            tmp_list = line.split(' ')
            for tmp in tmp_list:
                if punctuation.is_emoji(tmp):
                    add_cnt_dict(cnt_dict, "EMOJI", 1)
                if punctuation.is_number(tmp):
                    add_cnt_dict(cnt_dict, "NUMBER", 1)
                if tmp == ".":
                    add_cnt_dict(cnt_dict, "SENTENCE", 1)
                if tmp == 'rt':
                    add_cnt_dict(cnt_dict, "RT", 1)
            if len(tmp_list) > 60:
                add_cnt_dict(cnt_dict, "SIZE40", 1)

        print("get result from filename:", filename)
        print("cnt_dict:", cnt_dict)
        print("rate:")
        rate_dict = getRate(cnt_dict, len(total_list))
        file_rate_list.append([filename, rate_dict])
    print('file_rate_list:', file_rate_list)
    return file_rate_list

###取2个size40的进行区分
def common_sort(e, text):
    some_dict = e[1]
    if text in some_dict:
        return some_dict[text]
    else:
        return 0

def my_sort_size40(e):
    return common_sort(e, "SIZE40")
def my_sort_emoji(e):
    return common_sort(e, "EMOJI")
def my_sort_number(e):
    return common_sort(e, "NUMBER")
def my_sort_sentence(e):
    return common_sort(e, "SENTENCE")
def my_sort_rt(e):
    return common_sort(e, "RT")

if __name__ == '__main__':

    def isKika(file_dir):
        score = 0
        file_rate_list = get_file_rate_list(file_dir)
        file_rate_list.sort(key = my_sort_size40)
        print('file_rate_list my_sort_size40:', file_rate_list, '\n')

        file_rate_list.sort(key = my_sort_emoji)
        print('file_rate_list my_sort_emoji:', file_rate_list, '\n')

        file_rate_list.sort(key = my_sort_number)
        print('file_rate_list my_sort_number:', file_rate_list, '\n')

        file_rate_list.sort(key = my_sort_sentence)
        print('file_rate_list my_sort_sentence:', file_rate_list, '\n')

        file_rate_list.sort(key = my_sort_rt)
        print('file_rate_list my_sort_rt:', file_rate_list, '\n')
