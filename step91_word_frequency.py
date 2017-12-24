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

def add_cnt_dict(cnt_dict, k, v):
    if k in cnt_dict:
        v += cnt_dict[k]
    cnt_dict[k] = v

def getRate(cnt_dict, size):
    for k,v in cnt_dict.items():
        print('%-10s'%k, v, v/size)

if __name__ == '__main__':

    filename_list = tools.get_filename_list('raw_data/dir_step00')
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
            if len(tmp_list) > 40:
                add_cnt_dict(cnt_dict, "SIZE40", 1)

        print("get result from filename:", filename)
        print("cnt_dict:", cnt_dict)
        print("rate:")
        getRate(cnt_dict, len(total_list))