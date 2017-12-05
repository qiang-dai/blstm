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

def args():
    filename = 'raw_data/dir_step05'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    threshold_line_cnt = 10000
    if len(sys.argv) > 2:
        threshold_line_cnt = int(sys.argv[2])

    result_dir = 'raw_data/dir_step06'
    if len(sys.argv) > 3:
        res_file = sys.argv[3]
    return filename, threshold_line_cnt, result_dir

if __name__ == '__main__':
    file_dir, threshold_line_cnt, result_dir = args()

    print('file_dir, threshold_line_cnt, result_dir:', file_dir, threshold_line_cnt, result_dir )

    filename_list,_ = pyIO.traversalDir(file_dir)
    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]

    for file_index, filename in enumerate(filename_list):
        print('file_index, filename:', file_index, filename)
        sentences = pyIO.get_content(filename)

        parts = len(sentences)/threshold_line_cnt
        parts = int(parts)
        for i in range(parts+1):
            begin = i*threshold_line_cnt
            end = begin + threshold_line_cnt
            print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  'filename, i, begin, end, length:',
                   filename, i, begin, end, len(sentences[begin:end]))

            shortname = filename.split('/')[-1]
            dst_filename = result_dir + '/data_%02d_%s.txt'%(i, shortname)
            pyIO.save_to_file('\n'.join(sentences[begin:end]), dst_filename)