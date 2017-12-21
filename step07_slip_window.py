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


###读入所有内容
def line_to_train_text(sentence):# 以字符串的形式读入所有数据, 按行处理
    total_list = []

    punc_list = punctuation.get_punc_list()

    sentence = sentence.strip()
    tmp_list = sentence.split(' ')
    ###删除Header Tail
    if len(tmp_list) < 2:
        return total_list
    tmp_list = [e.split('/') for e in tmp_list]

    ###分类数据
    cat_type = tmp_list[0][0]
    ###填充
    category_prefix = (cat_type, 'SP', '')

    tmp_list = tmp_list[1:]
    ###头部填充
    cnt_fixed = int(punctuation.get_timestep_size()/2 - 1)
    for i in range(cnt_fixed):
        total_list.append((punctuation.get_filled_word(), punc_list[0], ''))
    ###合并当前
    total_list.extend(tmp_list)
    ###尾部填充
    cnt_fixed = int(punctuation.get_timestep_size()/2 - 1)
    for i in range(cnt_fixed):
        total_list.append((punctuation.get_filled_word(), punc_list[0], ''))

    res_list = []
    for i, item in enumerate(total_list):
        if i == 0:
            continue

        length = punctuation.get_timestep_size() - 1
        if i + length >= len(total_list):
            continue

        res_list.append([category_prefix,] + total_list[i:i+length])
    return res_list

def combine_line(filename, threshold_line_cnt, punc_list):
    # 以字符串的形式读入所有数据, 按行处理
    final_list = []

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "filename:", filename)
    sentences = pyIO.get_content(filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'sentences size:', len(sentences))

    #for sentence in sentences:
    for i in range(len(sentences)):
        if i%100 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "i, filename:", i, filename)
        if i > threshold_line_cnt:
            print("threshold_line_cnt:", threshold_line_cnt)
            break

        sentence = sentences[i]
        res_list = line_to_train_text(sentence)
        final_list.extend(res_list)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "len(final_list):", len(final_list))
    return final_list

###每隔 32 个单词就处理一下
def save_fixed_letter(filename, total_list, result_name, punc_list, file_index, result_dir, batch_pos):
    with open('data/word_tag_id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    res_list = []
    word_list = []
    label_list= []
    for index,tmp_list in enumerate(total_list):
        #print(filename, 'index,tmp_list:', index,tmp_list)

        w_list = [item[0] for item in tmp_list]
        p_list = [item[1] for item in tmp_list]

        ###用于debug
        res = ' '.join('/'.join(e) for e in tmp_list)

        w_id_list = [word2id[word] for word in w_list]
        p_id_list= [tag2id[punc] for punc in p_list]

        res_list.append(res)
        word_list.append(w_id_list)
        label_list.append(p_id_list)

    pyIO.save_to_file('\n'.join(res_list), result_name)

    res_list = []
    for w_list in word_list:
        #t_list = ['%d'%e for e in w_list]
        t_list = []
        for e in w_list:
            #print('e:', e)
            t_list.append('%d'%e)
        res_list.append(' '.join(t_list))
    pyIO.save_to_file('\n'.join(res_list), result_name.replace(".txt", '_word.txt'))

    res_list = []
    for p_list in label_list:
        t_list = ['%d'%e for e in p_list]
        res_list.append(' '.join(t_list))
    pyIO.save_to_file('\n'.join(res_list), result_name.replace(".txt", '_label.txt'))

    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),res_list[:30])

    ###写数据
    X = np.asarray(word_list)
    y = np.asarray(label_list)
    with open('%s'%(result_name.replace(".txt", "_data_patch_.pkl")), 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'save over')


if __name__ == '__main__':
    file_dir, threshold_line_cnt, result_dir = tools.args()

    filename_list,_ = pyIO.traversalDir(file_dir)
    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]
    filename_list.sort()

    for file_index, filename in enumerate(filename_list):

        short_filename = filename.split('/')[-1]
        short_filename = short_filename.replace('.txt.txt', '.txt')
        short_filename = short_filename.replace("step04_", "step07_")

        result_name = result_dir + '/' + short_filename

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename, threshold_line_cnt, result_name:', filename, threshold_line_cnt, result_name)

        punc_list = punctuation.get_punc_list()

        item_list = combine_line(filename, threshold_line_cnt, punc_list)

        #if len(item_list) > 500:
        if True:
            ###每100w条写一个文件
            for batch_pos in range(len(item_list)):
                beg_pos = batch_pos*100*10000
                end_pos = beg_pos + 100*10000
                if beg_pos >= len(item_list):
                    break
                print("beg_pos,end_pos:", beg_pos, end_pos)

                current_name = result_name.replace("step07_", "step07_%02d_"%batch_pos)

                print("file_index, filename:", file_index, filename)
                print("file_index, current_name:", file_index, current_name)

                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'save_fixed_letter')
                save_fixed_letter(filename, item_list[beg_pos:end_pos], current_name, punc_list, file_index, result_dir, batch_pos)
