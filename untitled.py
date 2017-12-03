# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print (sess.run(hello))

import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

pred=tf.placeholder(dtype=tf.bool,name='bool')
x = tf.placeholder(tf.float32, shape=(5, 1))
# y = tf.cond(pred,lambda:x,lambda:x-1)
#y = tf.cond(x > 0, lambda:x+1, lambda:x-1)
y = tf.where(x > 0, x + 1, x - 1)
z=tf.where(pred,x+1,x-1)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    rand_array = np.random.randn(5,1)
    print(rand_array)

    indices = [[0],[3]]
    g = tf.gather(rand_array, indices)
    print (sess.run(g))

    input = [[[1, 2, 3], [4, 5, 6]],
             [[13, 16, 14], [9, 8, 7]],
             [[18, 12, 15], [10, 17, 11]]]
    x = tf.reshape(input, [-1, 1])
    print ('x=\n', sess.run(x))

    y = tf.reshape(input, [-1])
    print ('y=\n', sess.run(y))

    indices = [[0,0],[0,3]]
    z = tf.gather(y, indices)
    print ('z=\n', sess.run(z))

    indices = [[0], [7]]
    z = tf.gather(y, indices)
    print ('z=\n', sess.run(z))

    w = tf.arg_max(input, 0)
    print('w=\n', sess.run(w))

    # y1,z1=sess.run([y,z],feed_dict={pred:True,  x:rand_array})
    # y2,z2=sess.run([y,z],feed_dict={pred:False, x:rand_array})
    # print('\ny1, z1:\n', y1, z1)
    # print('\ny2, z2:\n', y2, z2)
    #
    #
    # input = [[[1, 1, 1], [2, 2, 2]],
    #          [[3, 3, 3], [4, 4, 4]],
    #          [[5, 5, 5], [6, 6, 6]]]
    # print(tf.slice(input, [1, 0, 0], [1, 1, 3]))
    # tf.slice(input, [1, 0, 0], [1, 2, 3])
    #
    # tf.slice(input, [1, 0, 0], [2, 1, 3])
    #
    # print(tf.gather(input, [0, 2]))
    #
    # ### 对应维度的match
    # ###
    # #input =  [ 0,1 , 2 , 3 ]
    # #mask = np.array([True,False,True,False])
    # mask = [[[True, True, False], [True, True, False]],
    #         [[True, True, True], [False, True, False]],
    #         [[True, True, False], [True, False, False]]]
    # print(sess.run(tf.boolean_mask(input,mask)))
    #
    # print('gather')
    # input = [[1, 1, 1],
    #          [3, 3, 3],
    #          [5, 5, 5]]
    # indices = [[[0,1]]]
    # g = tf.gather(input, indices)
    # print (sess.run(g))

# x = tf.placeholder(tf.float32, shape=[None, 2])
# y_ = tf.placeholder(tf.float32, shape=[None, 2])
# loss = tf.reduce_sum(tf.abs(x))#Function chosen arbitrarily
# input_x=np.random.randn(100, 2)#Random generation of variable x
# input_y=np.random.randn(100, 2)#Random generation of variable y
#
# with tf.Session() as sess:
#     print(sess.run(loss, feed_dict={x: input_x, y_: input_y}))
