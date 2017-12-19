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

def format(line):
    line = line.strip()
    line = line.replace("\\" + "'" , "'")
    if len(line) < 10:
        return ''

    # if line.find('SmartRE') != -1:
    #     return ''
    # if line.find('smartre') != -1:
    #     return ''

    # if len(line.split(' ')) < 3:
    #     return ''
    # if len(line.split(' ')) > 30:
    #     return ''
    # if line[0] in punctuation.punctuation_all_list:
    #     return ''
    # if line[-2] in punctuation.punctuation_all_list:
    #     return ''
    #
    # if line[-1] in punctuation.punctuation_list:
    #     return line

    return line

def clean_sentence(line):
    line = line.replace("&amp;", ";")
    # line = line.replace(' : ', '.\n')
    # line = line.replace('. ', '.\n')
    # line = line.replace('? ', '?\n')
    # line = line.replace('! ', '!\n')
    ###判断点后是否大写
    pos = line.find('.')
    if pos != -1 and len(line) > pos+1 and punctuation.is_alphabet(line[pos+1]):
        if line[pos+1].upper() == line[pos+1]:
            res = line[:pos+1] + '\n' + line[pos+1]
            line = res

    line = line.replace('\\n', ' ')
    # tmp_list = line.split('\n')
    # tmp_list = [format(e) for e in tmp_list]
    # # tmp_list = [e for e in tmp_list if e.find('RT.') == -1]
    # tmp_list = [e for e in tmp_list if len(e) > 0]
    # tmp_list = [e for e in tmp_list if len(e) > 10]
    # tmp_list = [e for e in tmp_list if e.find('"') == -1]
    # tmp_list = [e for e in tmp_list if e.find('-') == -1]
    # tmp_list = [e for e in tmp_list if e.find('(') == -1]
    # tmp_list = [e for e in tmp_list if e.find(')') == -1]
    # tmp_list = [e for e in tmp_list if e.find('www.') == -1]
    # tmp_list = [e for e in tmp_list if e.find('http') == -1]
    # tmp_list = [e for e in tmp_list if e.find('/') == -1]
    #return tmp_list
    return [line,]


if __name__ == '__main__':
    file_dir = 'raw_data/dir_twitter'
    if len(sys.argv) > 1:
        file_dir = sys.argv[1]
    res_list = []
    filename_list = []

    threshold_line_cnt = 200*10000
    if len(sys.argv) > 2:
        threshold_line_cnt = int(sys.argv[2])

    if os.path.isfile(file_dir):
        filename_list.append(file_dir)
    else:
        filename_list,_ = pyIO.traversalDir(file_dir)

    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]
    filename_list.sort()
    print('filename_list:', filename_list)


    for filename in filename_list:
        dst_filename = "raw_data/dir_step00/cat2_" + filename.split('/')[-1].replace(".txt", "_res.txt")
        print('filename:', filename)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'dst_filename:', dst_filename)

        c_list = tools.get_total_limit_list(filename, threshold_line_cnt)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'len(c_list):', len(c_list))

        for i, c in enumerate(c_list):
            if i%10000 == 0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'i:', i)

            c = tools.add_space_inside_line(c)
            sub_list = clean_sentence(c)
            for sub in sub_list:
                #print(len(sub.split(' ')), sub)
                pass
            res_list.extend(sub_list)
            if len(res_list) > threshold_line_cnt:
                break

        pyIO.save_to_file("\n".join(res_list),  dst_filename)
        if len(res_list) > threshold_line_cnt:
            break




