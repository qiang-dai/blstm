import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os,sys
from collections import Counter
import punctuation

# threshold_line_cnt = 1000000
check_batch_size = 3
if len(sys.argv) > 1:
    check_batch_size = int(sys.argv[1])
mode_num = 6
if len(sys.argv) > 2:
    mode_num = int(sys.argv[2])
#
# # 以字符串的形式读入所有数据
# print (os.getcwd())
# with open('raw_data/res.txt', 'rb') as inp:
#     texts = inp.read().decode('utf8')
# sentences = texts.split('\n')  # 根据换行切分
# sentences = sentences[:threshold_line_cnt]
#
# def get_Xy(sentence):
#     """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
#     words_tags = re.findall('([\S]+)/([\S]+)', sentence)
#     if words_tags:
#         words_tags = np.asarray(words_tags)
#         words = words_tags[:, 0]
#         tags = words_tags[:, 1]
#         return words, tags # 所有的字和tag分别存为 data / label
#     return None
#
# datas = list()
# labels = list()
# print( 'Start creating words and tags data ...')
# for sentence in tqdm(iter(sentences)):
#     result = get_Xy(sentence)
#     if result:
#         datas.append(result[0])
#         labels.append(result[1])
#
# print( 'Length of datas is %d' % len(datas) )
# print( 'Example of datas: ', datas[0])
# print( 'Example of labels:', labels[0])
#
# df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
# #　句子长度
# df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
# df_data.head(2)

# 句子长度的分布
# import matplotlib.pyplot as plt
# df_data['sentence_len'].hist(bins=100)
# plt.xlim(0, 100)
# plt.xlabel('sentence_length')
# plt.ylabel('sentence_num')
# plt.title('Distribution of the Length of Sentence')
# plt.show()

# 1.用 chain(*lists) 函数把多个list拼接起来
# from itertools import chain
# all_words = list(chain(*df_data['words'].values))
# # 2.统计所有 word
# sr_allwords = pd.Series(all_words)
# sr_allwords = sr_allwords.value_counts()
# set_words = sr_allwords.index
# set_ids = range(1, len(set_words)+1) # 注意从1开始，因为我们准备把0作为填充值
# #tags = [ 'x', 's', 'b', 'm', 'e']
# tags = list(punctuation.punctuation_dict.keys())
# tags.insert(0, punctuation.punctuation_unkown)
# tag_ids = range(len(tags))
#
# # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
# word2id = pd.Series(set_ids, index=set_words)
# id2word = pd.Series(set_words, index=set_ids)
# tag2id = pd.Series(tag_ids, index=tags)
# id2tag = pd.Series(tags, index=tag_ids)
#
# vocab_size = len(set_words)
# print( 'vocab_size={}'.format(vocab_size))
#
# max_len = 32
# def X_padding(words):
#     """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
#     ids = list(word2id[words])
#     if len(ids) >= max_len:  # 长则弃掉
#         return ids[:max_len]
#     ids.extend([0]*(max_len-len(ids))) # 短则补全
#     return ids
#
# def y_padding(tags):
#     """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
#     ids = list(tag2id[tags])
#     if len(ids) >= max_len:  # 长则弃掉
#         return ids[:max_len]
#     ids.extend([0]*(max_len-len(ids))) # 短则补全
#     return ids
#
# df_data['X'] = df_data['words'].apply(X_padding)
# df_data['y'] = df_data['tags'].apply(y_padding)
#
# # 最后得到了所有的数据
# X = np.asarray(list(df_data['X'].values))
# y = np.asarray(list(df_data['y'].values))
# print( 'X.shape={}, y.shape={}'.format(X.shape, y.shape))
# print( 'Example of words: ', df_data['words'].values[0])
# print( 'Example of X: ', X[0])
# print( 'Example of tags: ', df_data['tags'].values[0])
# print( 'Example of y: ', y[0])

# 保存数据
import pickle
import os
#
# if not os.path.exists('data/'):
#     os.makedirs('data/')
#
# with open('data/data.pkl', 'wb') as outp:
#     pickle.dump(X, outp)
#     pickle.dump(y, outp)
#     pickle.dump(word2id, outp)
#     pickle.dump(id2word, outp)
#     pickle.dump(tag2id, outp)
#     pickle.dump(id2tag, outp)
# print( '** Finished saving the data.'   )
#导入数据
import pickle
with open('data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

# 划分测试集/训练集/验证集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
print( 'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

# ** 3.build the data generator
class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]

print( 'Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print( 'Finished creating the data generator.')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

'''
For Chinese word segmentation.
'''
# ##################### config ######################
decay = 0.85
max_epoch = 5
#max_max_epoch = 10
timestep_size = max_len = 32           # 句子长度
vocab_size = 5159    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = 27
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置


with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
         
def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)  
    
    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
  
    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)  
    
    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    # inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
#     try:
#         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     except Exception: # Old TensorFlow version only returns outputs not states
#         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ***********************************************************
    
    # ***********************************************************
    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state 
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
        
        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])
    # ***********************************************************
    return output # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')   
    
bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num]) 
    softmax_b = bias_variable([class_num]) 
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]
# correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))

# ***** 优化求解 *******
# tvars = tf.trainable_variables()  # 获取模型的所有参数
# grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
# optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器

# # 梯度下降计算
# train_op = optimizer.apply_gradients( zip(grads, tvars),
#     global_step=tf.contrib.framework.get_or_create_global_step())
# print( 'Finished creating the bi-lstm model.')
#
# def test_epoch(dataset):
#     """Testing or valid."""
#     _batch_size = 500
#     ### accuracy, cost 都是 op
#     fetches = [accuracy, cost]
#     _y = dataset.y
#     data_size = _y.shape[0]
#     batch_num = int(data_size / _batch_size)
#     if batch_num == 0:
#         return 0,0
#
#     start_time = time.time()
#     _costs = 0.0
#     _accs = 0.0
#     for i in range(batch_num):
#         X_batch, y_batch = dataset.next_batch(_batch_size)
#         feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
#         _acc, _cost = sess.run(fetches, feed_dict)
#         _accs += _acc
#         _costs += _cost
#     mean_acc= _accs / batch_num
#     mean_cost = _costs / batch_num
#     return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())
tr_batch_size = 128 
#max_max_epoch = 1000
display_num = 5  # 每个 epoch 显示是个结果
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
print('tr_batch_num:', tr_batch_num)
print('display_batch:', display_batch)

saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
last_10_acc = []
# for epoch in range(max_max_epoch):
#     _lr = 1e-4
#     if epoch > max_epoch:
#         _lr = _lr * ((decay) ** (epoch - max_epoch))
#     print( 'EPOCH %d， lr=%g' % (epoch+1, _lr))
#     start_time = time.time()
#     _costs = 0.0
#     _accs = 0.0
#     show_accs = 0.0
#     show_costs = 0.0
#     for batch in range(tr_batch_num):
#         fetches = [accuracy, cost, train_op]
#         X_batch, y_batch = data_train.next_batch(tr_batch_size)
#         feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5}
#         _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
#         _accs += _acc
#         _costs += _cost
#         show_accs += _acc
#         show_costs += _cost
#         if (batch + 1) % display_batch == 0:
#             valid_acc, valid_cost = test_epoch(data_valid)  # valid
#             print( '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
#                                                 show_costs / display_batch, valid_acc, valid_cost))
#             show_accs = 0.0
#             show_costs = 0.0
#     mean_acc = _accs / tr_batch_num
#     mean_cost = _costs / tr_batch_num
#     if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
#         save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
#         print( 'the save path is ', save_path)
#     print( '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
#     print( 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time)        )
#     if mean_acc > 0.999:
#         print ('mean_acc > 0.999')
#         break
#     last_10_acc.append(mean_acc)
#     if len(last_10_acc) > 10:
#         last_10_acc = last_10_acc[1:]
#     print('last_10_acc:', last_10_acc)
# # testing
# print( '**TEST RESULT:')
# test_acc, test_cost = test_epoch(data_test)
# print( '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost) )

# ** 导入模型
saver = tf.train.Saver()
best_model_path = 'ckpt/bi-lstm.ckpt-%d'%(mode_num)
saver.restore(sess, best_model_path)

# 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
X_tt, y_tt = data_train.next_batch(check_batch_size)
print( 'X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
print( 'X_tt = ', X_tt)
print( 'y_tt = ', y_tt)
feed_dict = {X_inputs:X_tt, y_inputs:y_tt, lr:1e-5, batch_size:check_batch_size, keep_prob:1.0}

### y_pred 是一个 op
fetches = [y_pred]
_y_pred = sess.run(fetches, feed_dict)

#,print(,'X_tt.shape=',,X_tt.shape,,'y_tt.shape=',,y_tt.shape)
print('X_tt=',X_tt)
print('y_tt=',y_tt)
print('_y_pred=',_y_pred)
print('_y_pred[0] size, shape:', _y_pred[0].size, _y_pred[0].shape)
print('X_tt, y_tt size:', X_tt.size, y_tt.size)
print('X_tt, y_tt shape:', X_tt.shape, y_tt.shape)

for i in range(check_batch_size):
    x = X_tt[i]

    length = len(x)
    beg = i*length
    end = (i+1)*length
    y = _y_pred[0][beg:end]

    x_index = [e for e in x if e > 0]
    y_index = [np.argmax(e) for e in y]
    print ("x:", x)
    #print ("y:", y)
    print ("x_index:", x_index)
    print ("y_index:", y_index)

    word_list = [id2word[e] for e in x_index]
    label_list =[id2tag[e] for e in y_index]
    print (word_list)
    print (label_list)

    res = ''
    for pos in range(len(word_list)):
        res += word_list[pos]
        if label_list[pos] != 'UNKOWN':
            res += label_list[pos]
        res += ' '

    res = res.replace('Header', '')
    res = res.replace(' Tail', '')
    print ('=====', res)

