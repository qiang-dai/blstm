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

threshold_word_cnt = 2

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
    ###新闻的处理
    sentence = sentence.replace('\n', ' ')
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

def transform_punc(w, punc_list, punc_set):
    ###如果符号连在一起, 就选择第一个符号
    if w not in punc_set:
        ###取最后的符号,忽略后面的符号
        if w.find('...') != -1:
            w = '...'
        else:
            w = w[-1]
        if w not in punc_set:
            ###实在奇怪的符号,就认为是空格
            w = punc_list[0]
            print('error: not punc:', w)
    ###可能多个符号在一起
    w = w.replace('/', 'LEFT')
    if w not in punc_set:
        w = punc_list[0]
    return w

def format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict, flag_ignore_complex_punc):
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))

    total_res_list = []
    for i in range(len(sentences)):
        if i%100 == 0:
            print('sentence total, i:',len(sentences), i)

        sentence = sentences[i]
        sub_sentence_list = sentence.split('\\n')
        for sub_sentence in sub_sentence_list:

            ###解析结果
            res_list = clean_sentence(sub_sentence)
            for res in res_list:
                add_cnt_dict(cnt_dict, res)

            #print(i, res_list)
            flag_combined_punc = False

            content_list = ['Header',]
            labels_list = [punc_list[0],]
            for w in res_list:
                ###如果是符号,就往前加
                if punctuation.is_punc(w[0]):
                    ###如果标点符号有2个连在一起，那么直接放弃这句
                    if w not in punc_set and flag_ignore_complex_punc:
                        flag_combined_punc = True
                        break

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
            ###如果标点符号有2个连在一起，那么直接放弃这句
            if flag_combined_punc and flag_ignore_complex_punc:
                print('ignore sentence:', sub_sentence)
                continue

            ###添加结束标记
            content_list.append('Tail')
            labels_list.append(punc_list[0])

            ###如果一个标点都没有，就忽略这句话
            if len(labels_list) == 1 \
                and labels_list[0] == punc_list[0]:
                continue

            out_list = []
            for pos in range(len(content_list)):
                punc = labels_list[pos]
                if punc == punc_list[0]:
                    pass
                else:
                    pass
                #text = content_list[pos] + '/' + punc
                text = [content_list[pos], punc]
                out_list.append(text)
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

    ###两遍过滤
    ###1,遍历所有,判断标点符号的全集
    ###2,根据标点符号全集,生成训练数据
    print('pyIO.get_content:', filename)

    ###换一种做法
    sentences = pyIO.get_content(filename)
    print('sentences length:', len(sentences))

    res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict, flag_ignore_complex_punc)
    print('res_list[:3]:', res_list[:3])

    ###统计平均值,然后过滤
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))
    print('res_list size:', len(res_list))
    total_cnt = 0
    for k in cleaned_punc_dict.keys():
        print(cleaned_punc_dict[k], '    ', k)
        total_cnt += cleaned_punc_dict[k]

    #avg_cnt = total_cnt/len(cleaned_punc_dict) *2/3
    ### 根据最低的标点符号来做阈值
    avg_cnt = get_min_punc_cnt(cleaned_punc_dict)

    # print('filtered...less than ', avg_cnt)
    # for k in cleaned_punc_dict.keys():
    #     if cleaned_punc_dict[k] < avg_cnt and len(k) > 1:
    #         print('ignore', cleaned_punc_dict[k], '    ', k)

    print('valid...large than ', avg_cnt)
    cnt = 0
    for k in cleaned_punc_dict.keys():
        if (cleaned_punc_dict[k] > avg_cnt and len(k) == 2) or len(k) == 1:
            cnt += 1
            print(cnt, 'valid', cleaned_punc_dict[k], '    ', k)
            # if k not in punc_set:
            #     punc_list.append(k)

    ###再次计算
    print ('cnt_dict size:', len(cnt_dict))
    ###词频统计
    cnt_dict = {
        'None' : threshold_word_cnt,
        'Header' : threshold_word_cnt,
        'Tail' : threshold_word_cnt,
    }
    punc_set = set(punc_list)
    print('punc_set size:', len(punc_set))
    ###存储punc列表

    res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict, flag_ignore_complex_punc)
    print('res_list[:3]:', res_list[:3])
    print('res_list size:', len(res_list))
    #print(' '.join(res_list))

    ###保存标点符号
    punctuation.save_punc_list(punc_list)

    ###保存最终处理结果,格式是:
    content_line_list = []
    punc_set = set(punc_list)

    ###统计标点符号的个数
    cnt_punc_dict = {}
    for punc in punc_list:
        cnt_punc_dict[punc] = 0

    for i, tmp_list in enumerate(res_list):
        line_list = []
        for word,punc in tmp_list:
            if word in cnt_dict and cnt_dict[word] < threshold_word_cnt:
                ###填充字符
                word = punctuation.get_filled_word()
            if punc not in punc_set:
                print('[%d] error punc:'%i, punc, tmp_list)
                punc = punc_list[0]

            cnt_punc_dict[punc] += 1

            line_list.append('%s/%s'%(word, punc))
        content_line_list.append(' '.join(line_list))

    ### Header/SP by/SP robert/SP browning/. Tail/SP
    pyIO.save_to_file('\n'.join(content_line_list), result_name)

    c = Counter(cnt_dict)
    print('add_cnt_dict:', len(cnt_dict), c.most_common(100))
    for k in cnt_dict.keys():
        print ('word', cnt_dict[k], k)

    # print ('cnt_dict size:', len(cnt_dict))
    print ('total word cnt:', get_total_cnt(cnt_dict, 0))
    # print ('total cnt >= 2 word cnt:', get_total_cnt(cnt_dict, 2))
    # print ('total cnt >= 5 word cnt:', get_total_cnt(cnt_dict, 5))
    print ('total cnt >= 10 word cnt:', get_total_cnt(cnt_dict, threshold_word_cnt))
    # print ('total cnt >= 20 word cnt:', get_total_cnt(cnt_dict, 20))

    for i, punc in enumerate(punc_list):
        print(i+1, punc, cnt_punc_dict[punc])

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



