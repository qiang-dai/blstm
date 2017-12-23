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

def get_more_text():
    word_list = ['cat0', 'cat1', 'cat2', 'cat3', 'NONE', 'Header', 'Tail', 'SP', 'EMOJI', 'rt']
    for i in range(10000):
        word_list.append('num%d'%i)

    res_list = []
    for i in range(5):
        label = '__label__%d'%i
        res_list.append(label + ' ' + ' '.join(word_list))
    return '\n'.join(res_list)

def get_word_vector():
    ###加载word embedding
    empty_embedding = [float(0.0) for i in range(100)]
    ###读入词映射
    with open('data/word_tag_id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    ###读入词向量
    c_list = pyIO.get_content('model_word.vec')

    embedding_dict = {}
    for c in c_list:
        t_list = c.split(' ')
        if len(t_list) != 101:
            print('error:', c)
        key = t_list[0]
        v_list = [float(e) for e in t_list[1:]]
        embedding_dict[key] = v_list

    ###处理
    final_matrix = []
    for i in range(len(id2word)+1):
        if i not in id2word:
            print ('i:', i)
            continue
        word = id2word[i]
        if word not in embedding_dict:
            print('get_word_vector len(empty_embedding):', len(empty_embedding))
            final_matrix.append(empty_embedding)
        else:
            v_list = embedding_dict[word]
            print('get_word_vector len(v_list):', len(v_list))
            final_matrix.append(v_list)
    print('len(final_matrix)', len(final_matrix))

    final_np =np.asarray(final_matrix, dtype=float)
    print('final_np', final_np.shape)
    return final_np

if __name__ == '__main__':

    result_filename = 'tmp/fastText_train.txt'
    pyIO.save_to_file("", result_filename)

    res_list = []

    filename_list = tools.get_filename_list('raw_data/dir_step00')
    ###每个目录取1000行
    for index, filename in enumerate(filename_list):

        label = step05_append_category.get_label_bye_filename(filename)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "index, file_dir, label:",
              index, filename, label)

        total_list = tools.get_total_limit_list(filename, 200000*10000)
        res_list.extend([label + e for e in total_list])
        print('result_filename:', result_filename)
        pyIO.append_to_file_nolock("\n".join(res_list) + '\n', result_filename)
    pyIO.append_to_file_nolock(get_more_text() + '\n', result_filename)


    ###生成wordVector
    model = fasttext.cbow(result_filename, 'model_word')
    print (model.words) # list of words in dictionary

    ###命令行
    #cmd = 'fastText-0.1.0/fasttext  supervised -input %s -output model'%(result_filename)
    #cmd = './fasttext predict model_classify.bin test.txt k'
    #subprocess.call(cmd, shell=True)
    classifier = fasttext.supervised(result_filename, 'model_classify', label_prefix='__label__')
    #model = fasttext.load_model('model_classify.bin')
    # res = classifier.predict('this is a try')
    res = step05_append_category.get_word_by_fastText("this is a try")
    print('res:', res)
    #
    res = step05_append_category.get_word_by_fastText('Frozen again. I hate that song')
    print('res:', res)
    #
    res = step05_append_category.get_word_by_fastText('Colonel George W. Taylor (later a Brigadier General and commander of the brigade until mortally wounded);')
    print('res:', res)


