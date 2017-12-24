import time
import os,sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pyIO
import step04_format_multi_punc
import punctuation
import pickle
from tensorflow.contrib import rnn
import datetime

### 设置显存根据需求增长
import tensorflow as tf
# config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np
import tools
import datetime
import subprocess
import fasttext
import step05_append_category
import numpy as np

def get_more_text():
    word_list = ['cat0', 'cat1', 'cat2', 'cat3', 'NONE', 'Header', 'Tail', 'SP', 'EMOJI', 'rt']
    for i in range(10000):
        word_list.append('num%d'%i)

    res_list = []
    for i in range(5):
        label = '__label__%d'%i
        res_list.append(label + ' ' + ' '.join(word_list))
    return '\n'.join(res_list)

if __name__ == '__main__':

    operation = sys.argv[1]
    test_file = sys.argv[2]
    if operation == 'train':
        result_filename = 'tmp/fastText_train.txt'
        if not os.path.exists(result_filename):
            pyIO.save_to_file("", result_filename)

            res_list = []

            filename_list = tools.get_filename_list('raw_data/dir_step00')
            ###每个目录取1000行
            for index, filename in enumerate(filename_list):

                label = step05_append_category.get_label_bye_filename(filename)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "index, file_dir, label:",
                      index, filename, label)

                total_list = tools.get_total_limit_list(filename, 2000*10000)
                res_list.extend([label + e for e in total_list])
                print('result_filename:', result_filename)
                pyIO.append_to_file_nolock("\n".join(res_list) + '\n', result_filename)
            pyIO.append_to_file_nolock(get_more_text() + '\n', result_filename)


        ###命令行(加朋）
        dim = 10
        lr = 0.005
        epoch = 1
        min_count = 1
        word_ngrams = 4
        bucket = 10000000
        thread = 8
        silent = 1


        ###命令行(默认）
        dim = 100
        lr = 0.05
        epoch = 5
        min_count = 5
        word_ngrams = 1
        bucket = 2000000
        thread = 12
        silent = 1

        label_prefix = '__label__'
        output_file = "model_classify"

        def run(lr, epoch, min_count, word_ngrams, bucket):
            fasttext.supervised(result_filename, output_file, lr=lr, epoch=epoch,min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,thread=thread, label_prefix=label_prefix)
            classifier = fasttext.load_model('model_classify.bin')

            test_file_list = ['valid_kika.txt',
                              'valid_subtitle.txt',
                              'valid_twitter.txt',
                              'valid_wiki.txt']

            for test_file in test_file_list:
                c_list = pyIO.get_content(test_file)
                big_text = ' '.join(c_list)
                labels = classifier.predict([big_text, ], 1)
                print('test_file, labels: ', test_file, labels)

        for lr in np.arange(0.005, 0.50, 0.1):
            for epoch in range(1, 6, 3):
                for min_count in range(1, 5, 2):
                    for word_ngrams in range(1, 6, 3):
                        #for bucket in range(200*10000, 1400*10000, 800*10000):
                        bucket = 1000*10000
                        print((lr, epoch, dim, min_count, word_ngrams, bucket))
                        run(lr, epoch, min_count, word_ngrams, bucket)
