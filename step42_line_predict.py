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
import BatchGenerator
import step08_line_window

### 设置显存根据需求增长
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

filename = 'p.txt'
orig_filename = 'tmp/line.txt'
###
step04_format_multi_punc.main(filename, 1000, 'tmp/', 1, False)

###step04
file_list,_ = pyIO.traversalDir('tmp/')
step04_list = [e for e in file_list if e.find('step04_') != -1]

###step08
punc_list = punctuation.get_punc_list()
item_list = step08_line_window.combine_line(step04_list[0], 1000000, punc_list)
step08_line_window.save_fixed_letter('', item_list, orig_filename, punc_list, 0, 'tmp/')

###train/predict
file_list,_ = pyIO.traversalDir('tmp/')
step08_list = [e for e in file_list if e.find('data_patch_') != -1]

with open(step08_list[0], 'rb') as inp:
    x_list = pickle.load(inp)
    y_list = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

c_list = pyIO.get_content(orig_filename)
z_list = []
for c in c_list:
    tmp_list = c.split(' ')
    tmp_list = [e.split('/')[-1] for e in tmp_list]
    z_list.append(tmp_list)
'''
For Chinese word segmentation.
'''
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
                     avg_weight_change: weight_change_list}
        _acc, _cost = sess.run(fetches, feed_dict)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'test %d _acc, _cost:'%epoch, _acc, _cost)
        _accs += _acc
        _costs += _cost
    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()  # 最多保存的模型数量

data_patch_filename_list,_ = pyIO.traversalDir("raw_data/dir_step08")
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

filename = sys.argv[1]
content_list = pyIO.get_content(filename)
punc_list = punctuation.get_punc_list()
cnt_dict = {}
cleaned_punc_dict= {}
total_res = pyIO.get_content(filename)

# 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'x_list = ', x_list)
feed_dict = {X_inputs:x_list, lr:1e-5, batch_size:len(x_list), keep_prob:1.0, total_size:2*punctuation.get_timestep_size()}

### y_pred 是一个 op
fetches = [y_pred]
_y_pred = sess.run(fetches, feed_dict)

with open('data/word_tag_id.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

for i in range(len(x_list)):
    x = x_list[i]
    length = len(x)
    beg = i*length
    end = (i+1)*length
    y = _y_pred[0][beg:end]
    z = z_list[i]

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
    orig_list= []
    end_pos = 1000
    for i,word in enumerate(word_list):
        res += word
        orig = z[i]
        tag = label_list[i]
        if word == 'Tail':
            end_pos = i

        if tag == 'SP':
            pass
        else:
            res += tag
            orig+= tag
        res += ' '

        if i<= end_pos:
            orig_list.append(orig)
    print ('predict res :', res)

    print ('predict orig:', ' '.join(orig_list))




