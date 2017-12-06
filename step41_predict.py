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

def get_fix_big_list(filename):
    ###读入所有的行
    content_list = pyIO.get_content(filename)
    ###去除所有的标点符号
    punc_list = punctuation.get_punc_list()
    cnt_dict = {}
    punc_dict= {}
    total_res = step04_format_multi_punc.format_content(content_list, punc_list, cnt_dict, punc_dict)
    print('total_res:', total_res)
    
    ###拼接起来
    big_list = []
    for i,sentence in enumerate(total_res):
        for index, item in enumerate(sentence):
            if item[0] == 'Header':
                if len(big_list) > 0 and big_list[-1][1] == punc_list[0]:
                    big_list[-1][1] = item[1]
                else:
                    pass
            elif item[0] == 'Tail':
                pass
            else:
                big_list.append(item)
    
    ###解析为id的形式
    ###1,前面添加15个置位，后面添加16个
    train_word_list = []
    ###头部填充
    cnt_fixed = int(punctuation.get_timestep_size()/2 - 1)
    for i in range(cnt_fixed):
        train_word_list.append([punctuation.get_filled_word(), punc_list[0], ''])
    ###中部
    for i,item in enumerate(big_list):
        train_word_list.append(item)
    ###尾部
    cnt_fixed = int(punctuation.get_timestep_size()/2)
    for i in range(cnt_fixed):
        train_word_list.append([punctuation.get_filled_word(), punc_list[0], ''])
    
    print('train_word_list:', train_word_list)
    return train_word_list

def get_train_id_list(train_word_list):
    with open('data/word_tag_id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    punc_list = punctuation.get_punc_list()

    ###组织成list，转为np.asarray
    x_list = []
    y_list = []
    for i, item in enumerate(train_word_list):
        end = i + punctuation.get_timestep_size()
        if end == len(train_word_list) + 1:
            break
        word_list = train_word_list[i:end]
        id_list = []
        tag_list= []
        for index,item in enumerate(word_list):
            word = item[0]
            tag = item[1]
            if word in word2id:
                id_list.append(word2id[word])
            else:
                id_list.append(0)
            if tag in tag2id:
                tag_list.append(tag2id[tag])
            else:
                tag_list.append(punc_list[0])
        x_list.append(id_list)
        y_list.append(tag_list)
    ###save
    ###写数据
    X = np.asarray(x_list)
    y = np.asarray(y_list)
    with open('data/data_predict.pkl', 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'save over')
    return x_list

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
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    inputs = tf.unstack(inputs, timestep_size, 1)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                     initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                               initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    print("output.get_shape():", output.get_shape())
    return output # [-1, hidden_size*2]
with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

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

def run(max_max_epoch, data_file, begin, end, x_list, train_word_list):
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
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    #    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Creating the data generator ...')
    #data_train = BatchGenerator.BatchGenerator(X_train, y_train, shuffle=True)
    #data_valid = BatchGenerator.BatchGenerator(X_valid, y_valid, shuffle=False)
    #data_test = BatchGenerator.BatchGenerator(X_test, y_test, shuffle=False)
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the data generator.')

    ### 设置显存根据需求增长
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.80

    sess = tf.Session(config=config)
    #if last_mode_index != 0:
    if True:
        # ** 导入模型
        saver = tf.train.Saver()
        ##这里找最新的ckpt模型
        filename_list,_ = pyIO.traversalDir('ckpt/')
        filename_list = [e for e in filename_list if e.find('.data-00000-of-00001') != -1]
        print('filename_list:', filename_list)

        if len(filename_list) > 0:
            def mysort(f):
                return time.ctime(os.path.getmtime(f))
            filename_list.sort(key = mysort)
            value = filename_list[-1]
            value = value.replace('.data-00000-of-00001','')
            value = value.split('-')[-1]

            best_model_path = 'ckpt/bi-lstm.ckpt-%s'%(value)
            print('best_model_path:', best_model_path)
            saver.restore(sess, best_model_path)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Finished creating the bi-lstm model.')

    sess.run(tf.global_variables_initializer())

    #size = data_train.X.shape[0]
    size = len(x_list)
    #X_tt, y_tt, offset, _, _, _ = data_train.next_batch(size)
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'X_tt = ', X_tt)
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'y_tt = ', y_tt)
    feed_dict = {X_inputs:x_list, lr:1e-5, batch_size:size, keep_prob:1.0, total_size:2*punctuation.get_timestep_size()}

    fetches = [y_pred]
    _y_pred = sess.run(fetches, feed_dict)


    res_list = []
    orig_list= []
    for i in range(size):
        x = x_list[i]
        ###每行去第15个内容和符号即可
        size = len(x)
        offset = int(size/2) - 1
        val = x[offset]
        word = punctuation.get_filled_word()
        if val in id2word:
            word = id2word[val]

        for pos in range(i*size,i*size + size):
            print(i, pos, np.argmax(_y_pred[0][pos]))

        tag_pos = np.argmax(_y_pred[0][i*size + offset])
        tag = id2tag[tag_pos]

        if tag == 'SP':
            tag = ''

        res_list.append(word + tag)

        tag = train_word_list[offset + i][1]
        if tag == 'SP':
            tag = ''
        orig_list.append(train_word_list[offset + i][2] + tag)

        # length = len(x)
        # beg = i*length
        # end = (i+1)*length
        # y = _y_pred[0][beg:end]
        #
        # x_index = [e for e in x if e > 0]
        # y_index = [np.argmax(e) for e in y]
        # print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"x:", x)
        # print ("x_index:", x_index)
        # print ("y_index:", y_index)
        #
        # word_list = [id2word[e] for e in x_index]
        # label_list =[id2tag[e] for e in y_index]
        # print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),word_list)
        # print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),label_list)
    print (' '.join(res_list))
    print (' '.join(orig_list))




if __name__ == '__main__':
    filename = "raw_data/dir_begin/4.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    train_word_list = get_fix_big_list(filename)
    x_list = get_train_id_list(train_word_list)
    print('x_list:', x_list)

    filename_list,_ = pyIO.traversalDir('data/')
    filename_list = [e for e in filename_list if e.find('data_patch_') != -1]
    #filename_list = filename_list[1:]
    print('filename_list:', filename_list)
    ###存储为数据


    run(2, 'data/data_predict.pkl', 0, 0, x_list, train_word_list)
