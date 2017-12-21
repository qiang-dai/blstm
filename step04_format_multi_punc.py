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
import pyString
import step05_append_category


###都使用小写字母
def transform(word):
    ###如果都是数字,就改为Num3 这种
    if punctuation.is_number(word[0]):
        word = 'num%d'%(len(word))
    elif punctuation.is_alphabet(word[0]):
        word = word.lower()
    elif punctuation.is_punc(word[0]):
        pass
        ###debug
        #add_cnt_dict(cleaned_punc_dict, word)
        #cleaned_punc_dict[word] = True
    elif punctuation.is_emoji(word[0]):
        word = 'EMOJI'
        return word
    else:
        pass
    return word.lower()

def merge_chars(word):
    #字符相连,数字相连,符号
    flag_list = [punctuation.getCharType(c) for c in word]
    ###合并同类项
    last = flag_list[0]
    res = ''
    res_list = []
    tmp_orig = []
    for i in range(len(flag_list)):
        cur = flag_list[i]
        ###合并
        if cur == last:
            res += word[i]
        ###另外的单词
        else:
            res_list.append(transform(res))
            tmp_orig.append(res)
            res = word[i]

        last = cur
    ###查错
    res_list.append(transform(res))
    tmp_orig.append(res)

    return res_list, tmp_orig

def clean_sentence(sentence):
    ###新闻的处理
    sentence = sentence.replace('\n', ' ')
    ###首先根据空格\空白字符进行处理
    sentence = re.sub('\s',' ',sentence)
    word_list = sentence.split(" ")
    res_list = []
    orig_list= []
    for word in word_list:
        word = word.strip()
        if len(word) == 0:
            continue
        tmp_list,tmp_orig = merge_chars(word)
        res_list.extend(tmp_list)
        orig_list.extend(tmp_orig)

    return res_list, orig_list

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

def format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict, fast_cat):
    punc_set = punctuation.get_punc_set()

    print('punc_set size:', len(punc_set))

    total_res_list = []
    for i in range(len(sentences)):
        if i%100000 == 0:
            print('sentence total, i:',len(sentences), i)

        sentence = sentences[i]
        if fast_cat is None:
            fast_cat = step05_append_category.get_word_by_fastText(sentence)

        sub_sentence = sentence.replace('\\n', ' ')
        ###解析结果
        res_list,res_orig = clean_sentence(sub_sentence)
        for res in res_list:
            add_cnt_dict(cnt_dict, res)

        ###过滤无标点句子
        flag_punc_find = False

        word_list = ['Header',]
        orig_list = ['']
        labels_list = [punc_list[0],]
        for i,w in enumerate(res_list):
            if punctuation.is_emoji(w[0]) \
                    or punctuation.is_alphabet(w[0]) \
                    or punctuation.is_number(w[0]):
                word_list.append(w)
                ###原始词语
                orig_list.append(res_orig[i])
                labels_list.append(punc_list[0])

            ###如果是符号,就往前加
            elif punctuation.is_punc(w[0]):
                flag_punc_find = True

                if w not in punc_set:
                    w = punctuation.get_punc_other()

                ###统计词频
                add_cnt_dict(cleaned_punc_dict, w)
                ###标点符号进行替换
                w = transform_punc(w, punc_list, punc_set)
                labels_list[-1] = w
            else:
                ###其他所有的不能识别的都认为是标点
                w = punctuation.get_punc_other()
                labels_list[-1] = w


        ###添加结束标记
        word_list.append('Tail')
        orig_list.append('')
        labels_list.append(punc_list[0])

        ###如果一个标点都没有，就忽略这句话
        if not flag_punc_find and fast_cat is None:
            continue

        out_list = []
        flag_valid_punc = False
        for pos, word in enumerate(word_list):
            ###这里加以替换：仅仅计算5个标点符号
            punc = labels_list[pos]
            if not punctuation.is_valid_punc(punc) \
                    and punc != 'SP':
                punc = punctuation.get_punc_other()
            else:
                flag_valid_punc = True

            out_list.append([word, punc, orig_list[pos]])
        ###无效的符号一律不训练
        if flag_valid_punc:
            out_list.insert(0, fast_cat)
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
            print('lost word limit:', limit, k, end=' ')
            break
    print('lost word limit cnt:', len(some_cnt_dict), end=' ')

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


    threshold_word_cnt = 10
    if len(sys.argv) > 4:
        threshold_word_cnt = int(sys.argv[4])

    return filename, threshold_line_cnt, res_file, threshold_word_cnt

def get_all_file_list(file_dir):
    filename_list = []
    if os.path.isfile(file_dir):
        filename_list.append(file_dir)
    else:
        filename_list,_ = pyIO.traversalDir(file_dir)

    filename_list.sort()
    print('filename_list:', filename_list)
    return filename_list

def main(file_dir, threshold_line_cnt, result_dir, threshold_word_cnt, flag_save_word_dict, use_fasttext):
    ###初始化:标点符号写文件
    punctuation.save_punc_list(punctuation.punctuation_list)

    ###所有文件列表
    filename_list = get_all_file_list(file_dir)

    ###词频统计:添加4个类别
    default_word_id_list = [
        (punctuation.get_filled_word(), 0),
        ('cat0',  1),
        ('cat1',  2),
        ('cat2',  3),
        ('cat3',  4),
        ('Header',5),
        ('Tail',  6),
        ('SP',    7)
    ]
    ###词字典
    cnt_dict = {}
    ###标点符号频率统计
    cleaned_punc_dict = {}

    ###映射表
    word2id = {}
    tag2id = {}
    id2word = {}
    id2tag = {}

    if not flag_save_word_dict:
        with open('data/word_tag_id.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
    else:
        ###添加分类的标记
        for k, v in default_word_id_list:
            word2id[k] = v
            id2word[v] = k

            cnt_dict[k] = threshold_word_cnt

    ###遍历写
    punc_list = punctuation.get_punc_list()
    for file_index, filename in enumerate(filename_list):
        ###重置结果文件
        result_filename = result_dir + '/step04_%02d_'%file_index + filename.split('/')[-1]
        print('file_index, filename:', file_index, filename)

        ###所有标点符号
        sentences = pyIO.get_content(filename)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'pyIO.get_content:', filename)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'sentences length:', len(sentences))

        if len(sentences) > threshold_line_cnt:
            sentences = sentences[:threshold_line_cnt]
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'sentences length:', len(sentences))

        punc_set = set(punc_list)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'punc_set size:', len(punc_set))

        ###词频统计
        punc_set = set(punc_list)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'punc_set size:', len(punc_set))
        ###存储punc列表
        if use_fasttext:
            fast_cat = None
        else:
            fast_cat = step05_append_category.get_word_by_filename(filename)

        res_list = format_content(sentences, punc_list, cnt_dict, cleaned_punc_dict, fast_cat)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'res_list[:3]:', res_list[:3])
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'res_list size:', len(res_list))

        ###保存标点符号
        punctuation.save_punc_list(punc_list)

        ###保存最终处理结果,格式是:
        content_line_list = []
        ###统计

        for i, tmp_list in enumerate(res_list):
            line_list = []
            #fast_cat = res_list[0]
            #tmp_list = tmp_list[1:]

            #for word,punc,orig in tmp_list:
            for complex_index, complex_item in enumerate(tmp_list):
                if complex_index == 0:
                    line_list.append(complex_item)
                    continue

                word,punc,orig = complex_item
                ###保存单词的时候，才需要替换为NONE
                if word in cnt_dict and cnt_dict[word] < threshold_word_cnt and flag_save_word_dict:
                    ###填充字符
                    word = punctuation.get_filled_word()

                ###对word进行打id, 如果是只读，就不写入
                if word not in word2id:
                    if flag_save_word_dict:
                        cnt = len(word2id)
                        word2id[word] = cnt
                        id2word[cnt] = word
                    else:
                        word = punctuation.get_filled_word()

                if punc not in punc_set:
                    print('[%d] error punc:'%i, punc, tmp_list)
                    punc = punc_list[0]

                line_list.append('%s/%s/%s'%(word, punc, orig))
            content_line_list.append(' '.join(line_list))

        ###保存文件
        pyIO.save_to_file('\n'.join(content_line_list) + '\n', result_filename)

        c = Counter(cnt_dict)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'add_cnt_dict:', len(cnt_dict), c.most_common(100))
        for k in cnt_dict.keys():
            print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'word', cnt_dict[k], k)

        #print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'total word cnt:', get_total_cnt(cnt_dict, threshold_word_cnt))
        #print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'total cnt >= 10 word cnt:', get_total_cnt(cnt_dict, threshold_word_cnt))

    ###存储word2id==========================================
    if flag_save_word_dict:
        ###映射表
        for i, punc in enumerate(punc_list):
            tag2id[punc] = i
            id2tag[i] = punc
        with open('data/word_tag_id.pkl', 'wb') as outp:
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(tag2id, outp)
            pickle.dump(id2tag, outp)

        vocab_size = len(word2id)
        print( 'vocab_size={}'.format(vocab_size))
        ###保存单词个数
        punctuation.save_word_cnt(vocab_size)

if __name__ == '__main__':

    file_dir, threshold_line_cnt, result_dir, threshold_word_cnt = get_args()
    use_fasttext_category = False
    main(file_dir, threshold_line_cnt, result_dir, threshold_word_cnt, True, use_fasttext_category)

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



