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
import step07_slip_window
import step51_fastText_classify

### 设置显存根据需求增长
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

filename = sys.argv[1]
orig_filename = 'tmp/line.txt'
###
use_fasttext = True
step04_format_multi_punc.main(filename, 100000, 'tmp/step04/', 1, False, use_fasttext)

###step04
file_list,_ = pyIO.traversalDir('tmp/step04/')
step04_list = [e for e in file_list if e.find('step04_') != -1]

###step08
punc_list = punctuation.get_punc_list()
item_list = step07_slip_window.combine_line(step04_list[0], 1000000, punc_list)
step07_slip_window.save_fixed_letter('', item_list, orig_filename, punc_list, 0, 'tmp/step07/', 0)

###train/predict
file_list,_ = pyIO.traversalDir('tmp/step07/')
step08_list = [e for e in file_list if e.find('data_patch_') != -1]

with open(step08_list[0], 'rb') as inp:
    x_list = pickle.load(inp)
    y_list = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

c_list = pyIO.get_content(orig_filename)

###格式化后的原始句子
nature_list = []

for c in c_list:
    tmp_list = c.split(' ')
    text_list = []
    for e in tmp_list:
        word,tag,orig = e.split('/')
        text = orig
        if tag != punctuation.get_punc_list()[0]:
            text += tag
        text_list.append(text)
    nature_list.append(' '.join(text_list))

def get_nature_word(i, j):
    c = c_list[i]
    tmp_list = c.split(' ')
    word,tag,orig = tmp_list[j].split('/')
    return orig

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
hidden_size = punctuation.get_batch_size()  # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 7  # 最大梯度（超过此值的梯度将被裁剪）

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

# def test_epoch(dataset, epoch):
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
#         X_batch, y_batch, offset, index_list, weight_change_list, batch_cnt_punc_dict = dataset.next_batch(_batch_size)
#         feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0,
#                      avg_offset:offset,
#                      total_size:_batch_size*punctuation.get_timestep_size(),
#                      avg_index_list: index_list,
#                      avg_weight_change: weight_change_list,
#                      embedding: word_embedding_vector}
#         _acc, _cost = sess.run(fetches, feed_dict)
#         print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'test %d(%d %d) _acc, _cost:'%(epoch, batch, tr_batch_num), _acc, _cost)
#         _accs += _acc
#         _costs += _cost
#     mean_acc= _accs / batch_num
#     mean_cost = _costs / batch_num
#     return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()  # 最多保存的模型数量

data_patch_filename_list,_ = pyIO.traversalDir("tmp/step07")
data_patch_filename_list = [e for e in data_patch_filename_list if e.find('data_patch_') != -1]
print('data_patch_filename_list:', data_patch_filename_list)


###下一轮循环使用上次的模型
def get_model_name():
    ##这里找最新的ckpt模型
    model_name,_ = pyIO.traversalDir('ckpt/')
    model_name = [e for e in model_name if e.find('.data-00000-of-00001') != -1]
    print('model_name:', model_name)

    if len(model_name) > 0:
        def mysort(f):
            return time.ctime(os.path.getmtime(f))
        model_name.sort(key = mysort)
        value = model_name[-1]
        value = value.replace('.data-00000-of-00001','')
        value = value.split('-')[-1]

        best_model_path = 'ckpt/bi-lstm.ckpt-%s'%(value)
        print('best_model_path:', best_model_path)
        return best_model_path, int(value)
    return '',-1

model_name, pos = get_model_name()
model_name, _ = get_model_name()
if len(model_name) > 0:
    saver.restore(sess, model_name)

# filename = sys.argv[1]
# content_list = pyIO.get_content(filename)
# punc_list = punctuation.get_punc_list()
# cnt_dict = {}
# cleaned_punc_dict= {}
# total_res = pyIO.get_content(filename)

###统计
cnt_punc_category_dict = {}
total_batch_cnt_punc_dict = {}
for i in range(len(punctuation.get_punc_list())):
    key = '%d'%i
    cnt_punc_category_dict[key] = {}
    cnt_punc_category_dict[key]['good']  = 0.1
    cnt_punc_category_dict[key]['bad']   = 0.1
    cnt_punc_category_dict[key]['error'] = 0.1

###
total_input = 0
total_good = 0
total_output = 0
class_0_good = 0
class_0_bad = 0
class_0_error = 0

total_res_list = []
# 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'x_list = ', x_list)
for batch_pos in range(len(x_list)):
    batch_pos_beg = batch_pos*500
    batch_pos_end = batch_pos_beg+500
    if batch_pos_beg >= len(x_list):
        break

    cur_list = x_list[batch_pos_beg:batch_pos_end]

    feed_dict = {X_inputs:cur_list,
                 lr:1e-5,
                 batch_size:len(cur_list),
                 keep_prob:1.0,
                 total_size:2*punctuation.get_timestep_size(),
                 embedding: word_embedding_vector}

    ### y_pred 是一个 op
    fetches = [y_pred]
    _y_pred = sess.run(fetches, feed_dict)

    with open('data/word_tag_id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    total_res = ''
    for index in range(len(cur_list)):
        flag_sentence_end = False

        focus_index = punctuation.get_timestep_size()/2
        focus_index = int(focus_index)

        x = cur_list[index]
        length = len(x)
        beg = index*length
        end = (index+1)*length
        y = _y_pred[0][beg:end]
        orig_y = y_list[batch_pos_beg+index]

        x_index = x
        y_index = [np.argmax(e) for e in y]
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"x:", x)
        print ("x_index:", x_index)
        print ("y_index:", y_index)

        word_list = [id2word[e] for e in x_index]
        label_list =[id2tag[e] for e in y_index]


        if x_index[focus_index+1] == word2id['Tail']:
            flag_sentence_end = True

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "word_list:", word_list)
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label_list:", label_list)

        res = ''
        for i,word in enumerate(word_list):
            if i != focus_index:
                continue
            if word == 'Tail':
                res += '\n'
                break

            res += get_nature_word(batch_pos_beg+index, i)
            tag = label_list[i]

            if tag != 'SP':
                res += tag
            res += ' '
            if flag_sentence_end:
                res += '\n'

            ###这里做统计
            key = '%d'%(orig_y[i])
            other_key = '%d'%(y_index[i])

            if key == other_key:
                cnt_punc_category_dict[key]['good'] += 1
            else:
                cnt_punc_category_dict[key]['bad'] += 1
                cnt_punc_category_dict[other_key]['error'] += 1

        total_res += res
        total_res += ' '
        #print ('predict_res :', res)
        #print ('predict_orig:', nature_list[index])
    print('total_res', total_res)
    total_res_list.append(total_res)
pyIO.save_to_file('\n'.join(total_res_list), "predict_result_%s.txt"%filename)
for i in range(len(punctuation.get_punc_list())):
    key = '%d'%i
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
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'召回率：', '%6f'%(cnt_good/cnt_input), '%6d'%cnt_good, '%6d'%cnt_input, end = ' ')
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




