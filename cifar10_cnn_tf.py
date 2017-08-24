import tensorflow as tf
import numpy as np
from model import *

import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def expand_label(label_list) :  # from (4,2,8,1,1,5, ...) -> ([0,0,0,1,0,0,0], [0,1,0,0,0,0,0], ...)
    l = len(label_list)
    result = np.zeros((l,10))
    for index, value in enumerate(label_list) : 
        result[index, value] = 1
    return result

data_batch_1 = unpickle('data_batch_1') # keys : b'data', b'labels'
image = data_batch_1[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
label = expand_label(data_batch_1[b'labels'])

test_data = unpickle('test_batch')
test_image = test_data[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
test_label = expand_label(test_data[b'labels'])


x = tf.placeholder("float", shape=[None, 32, 32, 3])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float", shape=())

pred, mid = wideResNet(x, symmetry='e', width_factor=1, depth_factor=5)

loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = pred)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # (l, 10)
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#optimizer = tf.train.AdamOptimizer(1e-10).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)




def train_one_epoch() : 
    batch_size = 100
    data_len = 10000
    repeat_num = data_len//batch_size
    
    from_index = 0
    to_index = batch_size
    for i in range(repeat_num) : 
        input = image[from_index:to_index]
        target = label[from_index:to_index]
        sess.run(optimizer, feed_dict = {x : input, y_ : target, keep_prob : 0.5})
        
        if i % 1 == 0 : 
            train_accuracy = sess.run(accuracy, feed_dict = {x : input, y_ : target, keep_prob : 1})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            #print(sess.run(mid, feed_dict = {x : input, y_ : target, keep_prob : 1}))
        
        from_index += batch_size
        to_index += batch_size


def test() : 
    print("test accuracy %g"% sess.run(
        accuracy, feed_dict={x: test_image, y_: test_label, keep_prob: 1.0}))
    return

def train(repeat_num) : 
    for _ in range(repeat_num) : 
        train_one_epoch()
        test()

#train(1)
train_one_epoch()

'''
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
print("test accuracy %g"% sess.run(
    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))'''
