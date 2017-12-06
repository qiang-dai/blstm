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

from tensorflow.contrib import rnn
import numpy as np

import tensorflow as tf
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

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell(keep_prob):
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

def bi_lstm(X_inputs, batch_size, keep_prob, embedding):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell(keep_prob) for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell(keep_prob) for _ in range(layer_num)], state_is_tuple=True)

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

def run(max_max_epoch, data_file, begin, end):
    ### 设置显存根据需求增长
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.Session(config=config)

    lr = tf.placeholder(tf.float32, [])
    keep_prob = tf.placeholder(tf.float32, [])
    avg_offset = tf.placeholder(tf.float32, [])
    avg_index_list = tf.placeholder(tf.int32, [None, 1])
    avg_weight_change = tf.placeholder(tf.float32, [None, timestep_size])
    total_size = tf.placeholder(tf.float32, [])
    batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
    model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置


    with tf.variable_scope('Inputs'):
        X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
        y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')
        print('X_inputs shape:', X_inputs.get_shape())
        print('y_inputs shape:', y_inputs.get_shape())
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

    bilstm_output = bi_lstm(X_inputs, batch_size, keep_prob, embedding)

    with tf.variable_scope('outputs'):
        softmax_w = weight_variable([hidden_size * 2, class_num])
        softmax_b = bias_variable([class_num])
        y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b
        print('softmax_w shape:', softmax_w.get_shape())
        print('softmax_b shape:', softmax_b.get_shape())
        print('y_pred shape:', y_pred.get_shape())

    ##这里找最新的ckpt模型
    model_name,_ = pyIO.traversalDir('ckpt/')
    model_name = [e for e in model_name if e.find('.data-00000-of-00001') != -1]
    print('model_name:', model_name)

    sess.run(tf.global_variables_initializer())
    if len(model_name) > 0:
        def mysort(f):
            return time.ctime(os.path.getmtime(f))
        model_name.sort(key = mysort)
        value = model_name[-1]
        value = value.replace('.data-00000-of-00001','')
        value = value.split('-')[-1]

        best_model_path = 'ckpt/bi-lstm.ckpt-%s'%(value)
        print('best_model_path:', best_model_path)
        saver = tf.train.Saver()
        saver.restore(sess, best_model_path)
    else:
        pass

    import pickle
    with open(data_file, 'rb') as inp:
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
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
        X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Creating the data generator ...')
    data_train = BatchGenerator.BatchGenerator(X_train, y_train, shuffle=True)
    data_valid = BatchGenerator.BatchGenerator(X_valid, y_valid, shuffle=False)
    data_test = BatchGenerator.BatchGenerator(X_test, y_test, shuffle=False)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the data generator.')


    '''
    For Chinese word segmentation.
    '''
    # adding extra statistics to monitor
    # y_inputs.shape = [batch_size, timestep_size]

    ###预测的结果
    ###--------------------------------------------------------------------------------------------------------
    ###--------------------------------------------------------------------------------------------------------
    ###--------------------------------------------------------------------------------------------------------
    ### y_pred是一个二维数组，包含128*32个元素（列）。 每一子维度是34（32个标点符号的类别），
    ### avg_index_list 很自然就是一维数组
    ###按行取最大值，然后截取索引
    y_result_item = tf.gather(tf.cast(tf.argmax(y_pred, 1), tf.int32), avg_index_list)
    ###输入的结果
    y_input_item = tf.gather(tf.reshape(y_inputs, [-1]), avg_index_list)
    correct_prediction = tf.equal(y_result_item, y_input_item)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #show_tensor= tf.Print(y_pred, [y_pred], message='y_pred')
    # show_tensor= y_pred.get_shape().as_list()

    #correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
    # accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # accuracy = (tf.cast(accuracy2, tf.float32)*total_size - avg_offset) / (total_size - avg_offset)

    #tf.div(tf.matmul(accuracy2, total_size) - avg_offset, total_size - avg_offset)
    res_pred_index = tf.reshape(tf.gather(y_pred, avg_index_list), [-1, class_num])

    show_tensor1 = y_input_item
    show_tensor2 = res_pred_index
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_input_item, [-1]), logits = res_pred_index))
    #cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=tf.reshape(y_pred, [-1, timestep_size, class_num]), targets=y_inputs, weights=avg_weight_change))

    # ***** 优化求解 *******
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器

    # 梯度下降计算
    train_op = optimizer.apply_gradients( zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the bi-lstm model.')

    def test_epoch(dataset):
        """Testing or valid."""
        _batch_size = 500
        ### accuracy, cost 都是 op
        fetches = [accuracy, cost]
        _y = dataset.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)
        if batch_num == 0:
            return 0,0

        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        for i in range(batch_num):
            X_batch, y_batch, offset, index_list, weight_change_list, batch_cnt_punc_dict = dataset.next_batch(_batch_size)
            feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0,
                         avg_offset:offset,
                         total_size:_batch_size*punctuation.get_timestep_size(),
                         avg_index_list: index_list,
                         avg_weight_change: weight_change_list}
            _acc, _cost = sess.run(fetches, feed_dict)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'test _acc, _cost:', _acc, _cost)
            _accs += _acc
            _costs += _cost
        mean_acc= _accs / batch_num
        mean_cost = _costs / batch_num
        return mean_acc, mean_cost

    #if last_mode_index != 0:
    #if begin != 0:
        # ** 导入模型

    tr_batch_size = punctuation.get_batch_size()
    #max_max_epoch = 1000
    display_num = 5  # 每个 epoch 显示是个结果
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
    display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'tr_batch_num:', tr_batch_num)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'display_batch:', display_batch)

    # last_10_acc = []

    for epoch in range(begin, end):

        ###1统计准确率
        y_result_list = []
        y_input_list = []
        cnt_punc_category_dict = {}
        total_batch_cnt_punc_dict = {}
        ###每一轮都重置
        for i in range(len(punctuation.get_punc_list())):
            key = '%d'%i
            cnt_punc_category_dict[key] = {}
            cnt_punc_category_dict[key]['input'] = 0.1
            cnt_punc_category_dict[key]['good']  = 0.1
            cnt_punc_category_dict[key]['bad']   = 0.1
            cnt_punc_category_dict[key]['error'] = 0.1

            total_batch_cnt_punc_dict[key] = 0

        _lr = 1e-4
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch - max_epoch))

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'EPOCH %d， lr=%g' % (epoch, _lr))
        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        show_accs = 0.0
        show_costs = 0.0
        for batch in range(tr_batch_num):
            fetches = [accuracy, cost, train_op, y_result_item, y_input_item, show_tensor1, show_tensor2]
            X_batch, y_batch, offset, index_list, weight_change_list, batch_cnt_punc_dict = data_train.next_batch(tr_batch_size)
            feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5,
                         avg_offset:offset,
                         total_size:tr_batch_size*punctuation.get_timestep_size(),
                         avg_index_list: index_list,
                         avg_weight_change: weight_change_list}
            _acc, _cost, _, predict_res, input_res, show_result1, show_result2 = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
            #print('show_result1:', show_result1)
            #print('show_result2:', show_result2)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'EPOCH %d, train _acc, _cost:'%epoch, _acc, _cost)
            y_result_list.append(predict_res)
            y_input_list.append(input_res)

            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                valid_acc, valid_cost = test_epoch(data_valid)  # valid
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                    show_costs / display_batch, valid_acc, valid_cost))
                show_accs = 0.0
                show_costs = 0.0
            ###统计标点符号出现次数
            for k in batch_cnt_punc_dict:
                total_batch_cnt_punc_dict[k] += batch_cnt_punc_dict[k]

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        if True:
            saver = tf.train.Saver()  # 最多保存的模型数量
            save_path = saver.save(sess, model_save_path, global_step=(epoch))
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'the save path is ', save_path, ', epoch:', epoch)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time)        )
        if mean_acc > 0.999:
            print ('mean_acc > 0.999')
            break
        # last_10_acc.append(mean_acc)
        # if len(last_10_acc) > 10:
        #     last_10_acc = last_10_acc[1:]
        # print('last_10_acc:', last_10_acc)
        ### 统计各个标点符号分类的结果
        for i in range(len(y_input_list)):
            tmp_input = y_input_list[i]
            tmp_result= y_result_list[i]
            for j in range(tmp_input.size):
                category = tmp_input[j][0]
                key = '%d'%category

                ###输入的标点符号个数
                cnt_punc_category_dict[key]['input'] += 1

                ###识别对的标点符号个数（召回）
                if tmp_input[j][0] == tmp_result[j][0]:
                    cnt_punc_category_dict[key]['good'] += 1
                else:
                    cnt_punc_category_dict[key]['bad'] += 1
                    ###其他的标点符号受影响了
                    other_key = '%d'%( tmp_result[j][0] )
                    cnt_punc_category_dict[other_key]['error'] += 1

        ###
        total_input = 0
        total_good = 0
        total_output = 0
        for i in range(len(punctuation.get_punc_list())):
            key = '%d'%i

            total_batch = total_batch_cnt_punc_dict[key]
            if (cnt_punc_category_dict[key]['good'] != 0 \
                or cnt_punc_category_dict[key]['bad'] != 0):
                ###待识别的结果总数
                cnt_input = cnt_punc_category_dict[key]['input']
                ###识别对的结果数
                cnt_good = cnt_punc_category_dict[key]['good']
                ###识别错的结果数
                cnt_bad = cnt_punc_category_dict[key]['bad']
                ###识别出错的结果
                cnt_error = cnt_punc_category_dict[key]['error']

                total_output += cnt_good
                total_output += cnt_error

                ###整体统计
                total_input += cnt_input
                total_good += cnt_good

                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, id2tag[i], end = ' ')
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'召回率：', '%6f'%(cnt_good/cnt_input), '%6d'%cnt_good, '%6d'%cnt_input, total_batch, end = ' ')
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'准确率：', '%6f'%(cnt_good/(cnt_good+cnt_error)), '%6d'%cnt_good, '%6d'%(cnt_good+cnt_error))
        ###整体准确率
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'整体召回率', total_good/total_input, total_good, total_input)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'整体准确率', total_good/total_output, total_good, total_output)

    # testing
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '**TEST RESULT:')
    test_acc, test_cost = test_epoch(data_test)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost) )

    # # ** 导入模型
    # saver = tf.train.Saver()
    # best_model_path = 'ckpt/bi-lstm.ckpt-%d'%( epoch )
    # print('best_model_path:', best_model_path)
    # saver.restore(sess, best_model_path)

    # 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
    X_tt, y_tt, offset, _, _, _ = data_train.next_batch(10)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt = ', X_tt)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'y_tt = ', y_tt)
    feed_dict = {X_inputs:X_tt, y_inputs:y_tt, lr:1e-5, batch_size:10, keep_prob:1.0, total_size:2*punctuation.get_timestep_size()}

    ### y_pred 是一个 op
    fetches = [y_pred]
    _y_pred = sess.run(fetches, feed_dict)

    #,print(,'X_tt.shape=',,X_tt.shape,,'y_tt.shape=',,y_tt.shape)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'X_tt=',X_tt)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'y_tt=',y_tt)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'_y_pred=',_y_pred)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'_y_pred[0] size, shape:', _y_pred[0].size, _y_pred[0].shape)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'X_tt, y_tt size:', X_tt.size, y_tt.size)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'X_tt, y_tt shape:', X_tt.shape, y_tt.shape)

    for i in range(10):
        x = X_tt[i]

        length = len(x)
        beg = i*length
        end = (i+1)*length
        y = _y_pred[0][beg:end]

        x_index = [e for e in x if e > 0]
        y_index = [np.argmax(e) for e in y]
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"x:", x)
        #print ("y:", y)
        print ("x_index:", x_index)
        print ("y_index:", y_index)

        word_list = [id2word[e] for e in x_index]
        label_list =[id2tag[e] for e in y_index]
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),word_list)
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),label_list)

        res = ''
        for i,word in enumerate(word_list):
            res += word
            tag = label_list[i]
            if tag == 'SP':
                pass
            else:
                res += tag
            res += ' '
        print ('predict res:', res)

if __name__ == '__main__':

    file_dir = 'raw_data/dir_step05'
    if len(sys.argv) > 1:
        file_dir = sys.argv[1]

    data_patch_filename_list,_ = pyIO.traversalDir(file_dir)
    model_name = [e for e in data_patch_filename_list if e.find('data_patch_') != -1]
    #data_patch_filename_list = data_patch_filename_list[1:]
    print('data_patch_filename_list:', model_name)

    for i, data_file in enumerate(data_patch_filename_list):
        begin = i*1
        end = begin + 1
        run(1, data_file, begin, end)