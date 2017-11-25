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


###遍历目录并写成大文件
def merge_dir_files(file_dir):
    file_list,_ = pyIO.traversalDir(file_dir, False, False)
    print ('file_list:', file_list)

    # 以字符串的形式读入所有数据, 按行处理
    total_list = []
    for filename in file_list:
        with open(filename, 'rb') as inp:
            sentences = pyIO.get_content(filename)
            for sentence in sentences:
                tmp_list = sentence.split('. ')
                tmp_list = [e.strip() + '.' for e in tmp_list]
                total_list.extend(tmp_list)
        print("total_list size:", len(total_list))
        #if len(total_list) > 100*1000:
        #   break
    pyIO.save_to_file('\n'.join(total_list), 'raw_data/total_english.txt')
    print (total_list[:300])

if __name__ == '__main__':
    file_dir = 'WorldEnglish/'
    if len(sys.argv) > 1:
        file_dir = sys.argv[1]

    merge_dir_files(file_dir)
