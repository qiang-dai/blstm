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


###读入所有内容,依次拼起来
def combine_line(filename, threshold_line_cnt, result_name):
    # 以字符串的形式读入所有数据, 按行处理
    total_list = []
    sentences = pyIO.get_content(filename)
    #for sentence in sentences:
    for i in range(len(sentences)):
        if i > threshold_line_cnt:
            break

        sentence = sentences[i]
        tmp_list = sentence.split(' ')
        ###删除Header Tail
        if len(tmp_list) == 0:
            continue
        if tmp_list[0].find('Header/') != -1:
            tmp_list = tmp_list[1:]
        if len(tmp_list) == 0:
            continue
        if tmp_list[-1].find('Tail/') != -1:
            tmp_list = tmp_list[:-1]

        total_list.extend(tmp_list)
    print("total_list size:", len(total_list))
    return total_list

###每隔 32 个单词就处理一下
def save_fixed_letter(total_list, result_name):
    line_list = []
    for i in range(len(total_list)):
        end = i + 32
        line_list.append(' '.join(total_list[i:end]) + '\n')

    pyIO.save_to_file('\n'.join(line_list), result_name)
    print (line_list[:30])

if __name__ == '__main__':

    ###<program> WorldEnglish 1000000 raw_data/total_english.txt
    filename, threshold_line_cnt, result_name = tools.args()

    item_list = combine_line(filename, threshold_line_cnt, result_name)
    save_fixed_letter(item_list, result_name)
