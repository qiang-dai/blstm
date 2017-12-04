import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os,sys
from collections import Counter
import punctuation
import datetime
import pyIO
import numpy as np
import pickle
import BatchGenerator
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# ##################### config ######################
decay = 0.85
max_epoch = 5
#max_max_epoch = 10
timestep_size = max_len = punctuation.get_timestep_size()           # 句子长度
vocab_size = punctuation.get_word_cnt()+1    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = len(punctuation.get_punc_list())
hidden_size = punctuation.get_batch_size()    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
avg_offset = tf.placeholder(tf.float32, [])
avg_index_list = tf.placeholder(tf.int32, [None, 1])
avg_weight_change = tf.placeholder(tf.float32, [None, timestep_size])
total_size = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置


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

    with tf.variable_scope('embedding'):
        embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

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
    inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                     initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                               initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ***********************************************************

    # ***********************************************************
    # ** 3. bi-lstm 计算（展开）
    # with tf.variable_scope('bidirectional_rnn'):
    #     # *** 下面，两个网络是分别计算 output 和 state
    #     # Forward direction
    #     outputs_fw = list()
    #     state_fw = initial_state_fw
    #     with tf.variable_scope('fw'):
    #         for timestep in range(timestep_size):
    #             if timestep > 0:
    #                 tf.get_variable_scope().reuse_variables()
    #             (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
    #             outputs_fw.append(output_fw)
    #
    #     # backward direction
    #     outputs_bw = list()
    #     state_bw = initial_state_bw
    #     with tf.variable_scope('bw') as bw_scope:
    #         inputs = tf.reverse(inputs, [1])
    #         for timestep in range(timestep_size):
    #             if timestep > 0:
    #                 tf.get_variable_scope().reuse_variables()
    #             (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
    #             outputs_bw.append(output_bw)
    #     # *** 然后把 output_bw 在 timestep 维度进行翻转
    #     # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
    #     outputs_bw = tf.reverse(outputs_bw, [0])
    #     # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
    #     output = tf.concat([outputs_fw, outputs_bw], 2)
    #     output = tf.transpose(output, perm=[1,0,2])
    #     output = tf.reshape(output, [-1, hidden_size*2])
    # ***********************************************************
    print("output.get_shape():", output.get_shape())
    return output # [-1, hidden_size*2]