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
import SegmentBatchGenerator
import pyIO
from sklearn.model_selection import train_test_split
import pickle
from random import shuffle
import step51_fastText_classify
import step05_append_category

input_dir = "raw_data/dir_step07"
if len(sys.argv) > 1:
    input_dir = sys.argv[1]

max_max_epoch = 1
if len(sys.argv) > 2:
    max_max_epoch = int(sys.argv[2])

print('max_max_epoch:', max_max_epoch)

### 设置显存根据需求增长
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
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
timestep_size = max_len = punctuation.get_timestep_size()           # 句子长度
vocab_size = punctuation.get_word_cnt()    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = 64
embedding_size = 100       # 字向量长度
class_num = len(punctuation.get_punc_list())
hidden_size = punctuation.get_batch_size()*2  # 隐含层节点数
layer_num = 3        # bi-lstm 层数
max_grad_norm = 5  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
avg_offset = tf.placeholder(tf.float32, [])
avg_index_list = tf.placeholder(tf.int32, [None, 1])
avg_weight_change = tf.placeholder(tf.float32, [None, timestep_size])
total_size = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置
embedding2 = tf.placeholder(tf.float32, [vocab_size, embedding_size], name='embedding')

###加载word embedding
word_embedding_vector = step51_fastText_classify.get_word_vector()
with tf.variable_scope('embedding'):
    #embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    embedding = tf.placeholder(tf.float32, [vocab_size, embedding_size], name='embedding')
    #embedding = tf.get_variable(name="embedding", shape=[vocab_size, embedding_size], initializer=tf.constant_initializer(word_embedding_vector), trainable=False)

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
    print("output.get_shape():", output.get_shape())
    return output # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')
    print('X_inputs shape:', X_inputs.get_shape())
    print('y_inputs shape:', y_inputs.get_shape())

bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b
    print('softmax_w shape:', softmax_w.get_shape())
    print('softmax_b shape:', softmax_b.get_shape())
    print('y_pred shape:', y_pred.get_shape())

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
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_input_item, [-1]), logits = res_pred_index))
cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=tf.reshape(y_pred, [-1, timestep_size, class_num]), targets=y_inputs, weights=avg_weight_change))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients( zip(grads, tvars),
                                      global_step=tf.contrib.framework.get_or_create_global_step())
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the bi-lstm model.')

def test_epoch(dataset, epoch):
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
                     avg_weight_change: weight_change_list,
                     embedding: word_embedding_vector}
        _acc, _cost = sess.run(fetches, feed_dict)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'test %d(%d %d) _acc, _cost:'%(epoch, batch, tr_batch_num), _acc, _cost)
        _accs += _acc
        _costs += _cost
    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()  # 最多保存的模型数量

data_patch_filename_list,_ = pyIO.traversalDir(input_dir)
data_patch_filename_list = [e for e in data_patch_filename_list if e.find('data_patch_') != -1]
###这里限制为3000万句（10个文件）
def limit_file_cnt(total_file_list, cat, limit_cnt):
    tmp_list = [e for e in total_file_list if e.find(cat) != -1]
    return tmp_list[:limit_cnt]
total_limit_file_list = []
total_limit_file_list.extend(limit_file_cnt(data_patch_filename_list, 'cat0', 10))
total_limit_file_list.extend(limit_file_cnt(data_patch_filename_list, 'cat1', 10))
total_limit_file_list.extend(limit_file_cnt(data_patch_filename_list, 'cat2', 10))
total_limit_file_list.extend(limit_file_cnt(data_patch_filename_list, 'cat3', 10))
data_patch_filename_list = total_limit_file_list

data_patch_filename_list.sort()
print('data_patch_filename_list:', data_patch_filename_list)

###下一轮循环使用上次的模型
def get_model_name(cat_type_prefix):
    ##这里找最新的ckpt模型
    model_name,_ = pyIO.traversalDir('ckpt/')
    model_name = [e for e in model_name if e.find('%s'%cat_type_prefix) != -1]
    model_name = [e for e in model_name if e.find('.data-00000-of-00001') != -1]
    print('model_name:', model_name)

    if len(model_name) > 0:
        def mysort(f):
            return time.ctime(os.path.getmtime(f))
        model_name.sort(key = mysort)
        value = model_name[-1]
        value = value.replace('.data-00000-of-00001','')
        value = value.split('-')[-1]

        best_model_path = 'ckpt/%s_bi-lstm.ckpt-%s'%(cat_type_prefix, value)
        print('best_model_path:', best_model_path)
        return best_model_path, int(value)
    return '',-1

for pathch_file_index,data_file in enumerate(data_patch_filename_list):
    ###cat0 cat1 cat2 cat3
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'data_file:', data_file)
    cat_type = step05_append_category.get_word_by_filename(data_file)
    model_name, pos = get_model_name(cat_type)

    if pathch_file_index < pos/max_max_epoch:
        continue

    with open(data_file, 'rb') as inp:
        X = pickle.load(inp)
        y = pickle.load(inp)
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
        X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

    fill_word_id = word2id[punctuation.get_filled_word()]
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Creating the data generator ...')
    data_train = SegmentBatchGenerator.SegmentBatchGenerator(X_train, y_train, shuffle=True, fill_word_id=fill_word_id)
    data_valid = SegmentBatchGenerator.SegmentBatchGenerator(X_valid, y_valid, shuffle=False, fill_word_id=fill_word_id)
    data_test = SegmentBatchGenerator.SegmentBatchGenerator(X_test, y_test, shuffle=False, fill_word_id=fill_word_id)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the data generator.')

    tr_batch_size = punctuation.get_batch_size()
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
    #max_max_epoch = 1000
    display_num = 5  # 每个 epoch 显示是个结果
    display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'tr_batch_num:', tr_batch_num)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'display_batch:', display_batch)

    # ** 导入模型
    #saver = tf.train.Saver()
    model_name, _ = get_model_name(cat_type)
    if len(model_name) > 0:
        saver.restore(sess, model_name)

    for epoch in range(pathch_file_index*max_max_epoch, (pathch_file_index+1)*max_max_epoch):
        ###1统计准确率
        y_result_list = []
        y_input_list = []
        cnt_punc_category_dict = {}
        total_batch_cnt_punc_dict = {}
        ###每一轮都重置
        for i in range(len(punctuation.get_punc_list())):
            key = '%d'%i
            cnt_punc_category_dict[key] = {}
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
                         avg_weight_change: weight_change_list,
                         embedding: word_embedding_vector}
            _acc, _cost, _, predict_res, input_res, show_result1, show_result2 = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
            #print('show_result1:', show_result1)
            #print('show_result2:', show_result2)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'EPOCH %d(%d %d), train _acc, _cost:'%(epoch, batch, tr_batch_num), _acc, _cost)
            y_result_list.append(predict_res)
            y_input_list.append(input_res)

            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                valid_acc, valid_cost = test_epoch(data_valid, epoch)  # valid
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                                                                                                       show_costs / display_batch, valid_acc, valid_cost))
                show_accs = 0.0
                show_costs = 0.0
            ###统计标点符号出现次数
            for k in batch_cnt_punc_dict:
                total_batch_cnt_punc_dict[k] += batch_cnt_punc_dict[k]

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        if True:# (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
            save_path = saver.save(sess, model_save_path.replace("bi-", "%s_bi-"%cat_type), global_step=(epoch))
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'the save path is ', save_path)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time)        )
        ### 统计各个标点符号分类的结果
        for i in range(len(y_input_list)):
            tmp_input = y_input_list[i]
            tmp_result= y_result_list[i]
            for j in range(tmp_input.size):
                category = tmp_input[j][0]
                key = '%d'%category

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
        class_0_good = 0
        class_0_bad = 0
        class_0_error = 0
        for i in range(len(punctuation.get_punc_list())):
            key = '%d'%i

            total_batch = total_batch_cnt_punc_dict[key]
            ###识别对的结果数
            cnt_good = cnt_punc_category_dict[key]['good']
            ###识别错的结果数
            cnt_bad = cnt_punc_category_dict[key]['bad']
            ###识别出错的结果
            cnt_error = cnt_punc_category_dict[key]['error']

            if i == 0:
                class_0_good = cnt_good
                class_0_bad = cnt_bad
                class_0_error = cnt_error

            cnt_input = cnt_good + cnt_bad
            cnt_output= cnt_good + cnt_error

            ###整体统计
            total_output += cnt_output
            total_input += cnt_input
            total_good += cnt_good

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, id2tag[i], end = ' ')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'召回率：', '%6f'%(cnt_good/cnt_input), '%6d'%cnt_good, '%6d'%cnt_input, total_batch, end = ' ')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'准确率：', '%6f'%(cnt_good/cnt_output), '%6d'%cnt_good, '%6d'%cnt_output)
        ###整体准确率
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'整体召回率', total_good/total_input, total_good, total_input)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'整体准确率', total_good/total_output, total_good, total_output)

        ###非空格整体准确率
        punc_good = total_good - class_0_good
        punc_intput = total_input - class_0_good - class_0_bad
        punc_output = total_output - class_0_good - class_0_error
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'非空格整体召回率', punc_good/punc_intput, punc_good, punc_intput)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'非空格整体准确率', punc_good/punc_output, punc_good, punc_output)
    # testing
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '**TEST RESULT:')
    test_acc, test_cost = test_epoch(data_test, epoch)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost) )

    # ** 导入模型
    saver = tf.train.Saver()
    print("pathch_file_index:", pathch_file_index)
    saver.restore(sess, get_model_name(cat_type)[0])

    # 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
    X_tt, y_tt, offset, _, _, _ = data_train.next_batch(10)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt = ', X_tt)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'y_tt = ', y_tt)
    feed_dict = {X_inputs:X_tt, y_inputs:y_tt, lr:1e-5, batch_size:10, keep_prob:1.0, total_size:2*punctuation.get_timestep_size()
        ,embedding: word_embedding_vector}

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
        print ("x_index:", x_index)
        print ("y_index:", y_index)

        word_list = [id2word[e] for e in x_index]
        label_list =[id2tag[e] for e in y_index]
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "word_list:", word_list)
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label_list:", label_list)

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


