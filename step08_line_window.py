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
import pickle

'''
按行处理文本
'''
###读入所有内容,按长度32进行拆分
def combine_line(filename, threshold_line_cnt, punc_list):
    # 以字符串的形式读入所有数据, 按行处理
    total_list = []

    sentences = pyIO.get_content(filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'sentences size:', len(sentences))

    #for sentence in sentences:
    timestep_size = punctuation.get_timestep_size()
    fix_word = punctuation.get_filled_word()
    fix_punc = punctuation.get_punc_list()[0]

    for i in range(len(sentences)):
        if i > threshold_line_cnt:
            break

        sentence = sentences[i]
        sentence = sentence.strip()
        tmp_list = sentence.split(' ')
        ###保留Header, 要Tail
        if len(tmp_list) <= 2:
            continue
        #tmp_list = tmp_list[:-1]

        ###如果满足32个词，如果大于32个词，就滑动窗口拆分为多句
        diff = len(tmp_list) - timestep_size
        if diff <= 0:
            fix_list = [fix_word + '/' + fix_punc + '/' for v in range(-1*diff)]
            tmp_list.extend(fix_list)
            total_list.append(tmp_list)
        else:
            for i in range(diff):
                cur_list = tmp_list[i:i+timestep_size]
                total_list.append(cur_list)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"total_list size:", len(total_list))
    return total_list

###每隔 32 个单词就处理一下
def save_fixed_letter(filename, total_list, result_name, punc_list, file_index, result_dir):
    line_list = []
    word_list = []
    label_list= []

    with open('data/word_tag_id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    for i, tmp_list in enumerate(total_list):
        if i %100000 == 0:
            print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'i:', i, len(total_list))

        res = []
        tmp_word_list = []
        tmp_label_list= []
        for index,item in enumerate(tmp_list):
            ###写文件
            res.append(item)

            ###2个对应的表
            # print('item:', item)
            tt_list = item.split('/')
            if len(tt_list) == 2:
                i = 0

            word,punc,orig = item.split('/')
            if word not in word2id:
                cnt = len(word2id) + 1
                word2id[word] = cnt
                id2word[cnt] = word

            tmp_word_list.append(word2id[word])
            tmp_label_list.append(tag2id[punc])

        line_list.append(' '.join(res))
        word_list.append(tmp_word_list)
        label_list.append(tmp_label_list)

    pyIO.save_to_file('\n'.join(line_list), result_name)
    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),line_list[:30])

    ###写数据
    X = np.asarray(word_list)
    y = np.asarray(label_list)
    with open('%s/data_patch_%02d.pkl'%(result_dir, file_index), 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'save over')
    return word_list, label_list

if __name__ == '__main__':
    file_dir, threshold_line_cnt, result_dir = tools.args()

    filename_list,_ = pyIO.traversalDir(file_dir)
    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]

    for file_index, filename in enumerate(filename_list):
        short_filename = filename.split('_')[-1]
        short_filename = short_filename.split('/')[-1]
        short_filename = short_filename.replace('.txt.txt', '.txt')

        result_name = result_dir + '/step08_%d_'%file_index + short_filename
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename, threshold_line_cnt, result_name:', filename, threshold_line_cnt, result_name)

        punc_list = punctuation.get_punc_list()

        item_list = combine_line(filename, threshold_line_cnt, punc_list)

        save_fixed_letter(filename, item_list, result_name, punc_list, file_index, result_dir)
