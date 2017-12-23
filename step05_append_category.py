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
import datetime
import tools
import tools
import fasttext


def get_word_by_filename(filename):
    for i in range(10):
        if filename.find('_cat%d_'%i) != -1:
            return 'cat%d'%i
    return 'xxx'

classifier = None
def get_word_by_fastText(text):
    global classifier
    if classifier is None:
        classifier = fasttext.load_model('model_classify.bin')
    labels = classifier.predict([text, ], 1)
    word = 'cat%s'%(labels[0][0].replace('__label__', ''))
    #print('get_word_by_fastText labels:', labels, ', word:', word, ', text:', text)

    return word
def get_word_by_vote(c_list):
    word_list = [get_word_by_fastText(e) for e in c_list]
    cnt_dict = {}
    for w in word_list:
        if w not in cnt_dict:
            cnt_dict[w] = 1
        else:
            cnt_dict[w] = cnt_dict[w]+1

    print('cnt_dict:', cnt_dict)
    ###max
    max_key = ''
    max_val = 0
    for k,v in cnt_dict.items():
        if v > max_val:
            max_key = k
            max_val = v
    return max_key

def get_label_bye_filename(filename):
    for i in range(10):
        if filename.find('cat%d_'%i) != -1:
            return '__label__%d , '%i
    return 'xxx'

def save_file_by_cat_filename(filename, i):
    dst_filename = "raw_data/dir_step05/step05_res_%02d"%i + filename.split('/')[-1].replace(".txt", "_res.txt")

    res_list = []
    c_list = pyIO.get_content(filename)
    word = get_word_by_filename(filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename:', filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'dst_filename:', dst_filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'word:', word, '\n')

    for c in c_list:
        res = word + ' ' + c
        res_list.append(res)

    pyIO.save_to_file("\n".join(res_list),  dst_filename)

def save_file_by_cat_fasttext(filename, i):
    dst_filename = "tmp/step05/step05_res_%02d"%i + filename.split('/')[-1].replace(".txt", "_res.txt")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename:', filename)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'dst_filename:', dst_filename)

    c_list = pyIO.get_content(filename)
    word = get_word_by_vote(c_list)

    print("filename, word:", filename, word)

if __name__ == '__main__':

    filename_list = tools.get_filename_list('raw_data/dir_step00')
    filename_list.sort()
    print('filename_list:', filename_list)
    operation = 'train'
    if len(sys.argv) > 1:
        filename_list = tools.get_filename_list(sys.argv[1])
    if len(sys.argv) > 2:
        operation = sys.argv[2]

    for i,filename in enumerate(filename_list):
        print('i, filename:', i, filename)
        
        if operation == 'test':
            save_file_by_cat_fasttext(filename, i)
        elif operation == 'train':
            save_file_by_cat_filename(filename, i)
        else:
            print('error operation:', operation)






