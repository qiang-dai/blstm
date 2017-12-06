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


###读入所有内容,依次拼起来
def combine_line(filename, threshold_line_cnt, punc_list):
    # 以字符串的形式读入所有数据, 按行处理
    total_list = []

    ###头部填充
    cnt_fixed = int(punctuation.get_timestep_size()/2 - 1)
    for i in range(cnt_fixed):
        total_list.append(punctuation.get_filled_word()+'/' + punc_list[0] + '/')

    sentences = pyIO.get_content(filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'sentences size:', len(sentences))

    #for sentence in sentences:
    for i in range(len(sentences)):
        if i > threshold_line_cnt:
            break

        sentence = sentences[i]
        sentence = sentence.strip()
        tmp_list = sentence.split(' ')
        ###删除Header Tail
        if len(tmp_list) == 0:
            continue
        if tmp_list[0].find('Header/') != -1:
            ###如果前面有标点符号，就需要添加到最后一个词的末尾
            if len(total_list) > 0:
                cur_punc = tmp_list[0].split('/')[1]

                last_word, last_punc, _ = total_list[-1].split('/')
                ###添加
                if cur_punc != punc_list[0] and last_punc == punc_list[0]:
                    total_list[-1] = last_word + '/' + cur_punc + '/' + last_word

            tmp_list = tmp_list[1:]
        if len(tmp_list) == 0:
            continue
        if tmp_list[-1].find('Tail/') != -1:
            tmp_list = tmp_list[:-1]

        total_list.extend(tmp_list)

    ###尾部填充
    punc_list = punctuation.get_punc_list()
    cnt_fixed = int(punctuation.get_timestep_size()/2)
    for i in range(cnt_fixed):
        total_list.append(punctuation.get_filled_word()+'/' + punc_list[0] + '/')
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

    for i in range(len(total_list)):
        if i%100000 == 0:
            print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'i:', i, len(total_list))

        end = i + punctuation.get_timestep_size()
        ###最后32个字符
        if end == len(total_list):
            break
        #line_list.append(' '.join(total_list[i:end]) + '\n')
        ###这里只取第 N/2 - 1 个标点符号进行预测
        res = []

        tmp_word_list = []
        tmp_label_list= []
        for index,item in enumerate(total_list[i:end]):
            #print(filename, 'item:', item)
            word,punc, _ = item.split('/')
            if index == punctuation.get_timestep_size()/2 - 1:
                res.append(item)
            else:
                res.append(word + '/' + punc_list[0] + '/' + word)

            ###2个对应的表
            if word not in word2id:
                cnt = len(word2id) + 1
                word2id[word] = cnt
                id2word[cnt] = word

            tmp_word_list.append(word2id[word])
            if index == punctuation.get_timestep_size()/2 - 1:
                tmp_label_list.append(tag2id[punc])
            else:
                tmp_label_list.append(tag2id[punc_list[0]])

        line_list.append(' '.join(res))
        word_list.append(tmp_word_list)
        label_list.append(tmp_label_list)

    pyIO.save_to_file('\n'.join(line_list), result_name)

    res_list = []
    for w_list in word_list:
        t_list = ['%d'%e for e in w_list]
        res_list.append(' '.join(t_list))
    pyIO.save_to_file('\n'.join(res_list), result_name.replace(".txt", '_word.txt'))

    res_list = []
    for w_list in label_list:
        t_list = ['%d'%e for e in w_list]
        res_list.append(' '.join(t_list))
    pyIO.save_to_file('\n'.join(res_list), result_name.replace(".txt", '_label.txt'))

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


if __name__ == '__main__':
    file_dir, threshold_line_cnt, result_dir = tools.args()

    filename_list,_ = pyIO.traversalDir(file_dir)
    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]

    for file_index, filename in enumerate(filename_list):
        short_filename = filename.split('_')[-1]
        short_filename = short_filename.split('/')[-1]
        short_filename = short_filename.replace('.txt.txt', '.txt')

        result_name = result_dir + '/step07_%d_'%file_index + short_filename
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename, threshold_line_cnt, result_name:', filename, threshold_line_cnt, result_name)

        punc_list = punctuation.get_punc_list()

        item_list = combine_line(filename, threshold_line_cnt, punc_list)

        save_fixed_letter(filename, item_list, result_name, punc_list, file_index, result_dir)
