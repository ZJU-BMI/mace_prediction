

from __future__ import print_function
import pandas as pd
import os
import tensorflow as tf
import math
from tensorflow.contrib import rnn
import numpy as np
import  random


'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''



def readTemDataCSV(time_steps,basepath,filename):
    data_source = pd.read_csv(base_path + filename, encoding='GBK')  # 读入时序数据
    data_source_list=data_source.iloc[:,:].values.tolist()
    x_data={}
    label = {}
    currentDataUnit = []
    currentPatientId = data_source_list[0][0]
    data_size=0
    for line in data_source_list:
        if line[0]==currentPatientId:
            if line[1]<time_steps-1:
                currentDataUnit.append(line[3:])
            elif line[1]==time_steps-1:
                currentDataUnit.append(line[3:])
                x_data.update({currentPatientId:currentDataUnit})
                if(line[2]==1):
                    label.update({currentPatientId:[0,1]})
                else:
                    label.update({currentPatientId: [1, 0]})
                data_size=data_size+1
        else:
            currentPatientId=line[0]
            currentDataUnit=[]
            currentDataUnit.append(line[3:])
    print("data1 loaded")
    return x_data,label

def readStaticData(basepath,filename,dic):
    data_source = pd.read_csv(base_path + filename, encoding='GBK')  # 读入时序数据
    data_source_list=data_source.iloc[:,:].values.tolist()
    data_size_sta = 0
    x_data={}
    skip = 0
    for line in data_source_list:
        if skip==0:
            skip=skip+1
            continue
            #跳过第一行
        if dic.__contains__(line[0]):
            x_data.update({line[0]:line[2:]})
            data_size_sta = data_size_sta+1
    print("static data loaded")
    return x_data

def BiRNNAndStatic(xTem ,xStatic , weights, biases,step,n_hidden):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(xTem, step, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    out = tf.matmul(outputs[-1], weights[0]) + biases[0]
    concat = tf.concat(axis=1,values=[out,xStatic])
    unnormalPred=tf.matmul(concat,weights[1])

    return unnormalPred

# Parameters
learning_rate = 0.001
#由于数据也不是很多。我们可以直接使用全数据训练，不需要用随机分组，也就不存在Batch Size这种设定了

base_path=os.path.abspath(os.curdir)+'/../../resources/'
filename1 = "output_5.0E-4_1866.csv"
print(filename1)
filename2 = "static_features.csv"
# Network Parameters
n_input = 1866 # 测试数据有维度
n_classes = 30 # 输出特征长度
n_static = 30 #静态数据维度
L2=0.1

time_step = 12
iteration = 10000
n_hidden = 64
display_step = 100
xTem = tf.placeholder("float", [None, time_step, n_input])
xSta = tf.placeholder("float", [None, n_static])
y = tf.placeholder("float", [None, 2])
# tf Graph input 数据输入项


# Launch the graph
x_tem_ori, y_ori = readTemDataCSV(time_step, base_path, filename1)
x_sta_ori = readStaticData(base_path, filename2, x_tem_ori)

x1 = []
x2 = []
y1 = []
for key in x_tem_ori:
    if x_sta_ori.__contains__(key):
        x1.append(x_tem_ori.get(key))
        x2.append(x_sta_ori.get(key))
        y1.append(y_ori.get(key))
print('x1 size = ', len(x1), ' x2 size = ', len(x2), ' y1 size = ', len(y1))
if len(x1) != len(x2) or len(x1) != len(y1) or len(y1) != len(x2):
    print('data size error')
print('hidden units = ', n_hidden, ', iteration = ', iteration, ' time step = ', time_step, ' data size = ',len(x1),'L2 = ',L2)

test_size=math.floor(len(x1) * 0.7)
x_tem_test=[]
y_test=[]
x_static_test=[]

train_data_size = math.floor(len(x1)*0.7)
currentSize = 0
for i in range(train_data_size):
    index = random.randint(0,train_data_size-currentSize)
    x_tem_test.append(x1[index][:])
    y_test.append(y1[index][:])
    x_static_test.append(x2[index][:])
    currentSize=currentSize+1
    x1.pop(index)
    x2.pop(index)
    y1.pop(index)
batch_tem_x = np.array(x_tem_test)
batch_sta_x = np.array(x_static_test)
batch_y = np.array(y_test)
test_tem_data = np.array(x1)
test_sta_data = np.array(x2)
test_label = np.array(y1)

# Define weights
weights = [
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    tf.Variable(tf.random_normal([2 * n_hidden, n_classes])),
    #30是静态数据维度
    tf.Variable(tf.random_normal([n_classes+30, 2]))
]
biases = [
    tf.Variable(tf.random_normal([n_classes]))
]
pred = BiRNNAndStatic(xTem, xSta, weights, biases,time_step,n_hidden)

#    Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + L2 * tf.nn.l2_loss(
        weights[0]) + L2 * tf.nn.l2_loss(weights[1])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
a = tf.cast(tf.argmax(pred, 1),tf.float32)
b = tf.cast(tf.argmax(y,1),tf.float32)
auc = tf.contrib.metrics.streaming_auc(a,b)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    step = 1
    # Keep training until reach max iterations

    while step <= iteration:
        # Run optimization  op (backprop)
        sess.run(optimizer, feed_dict={xTem: batch_tem_x, xSta: batch_sta_x, y: batch_y})

        if step % display_step == 0:
            accTrain = sess.run(accuracy, feed_dict={xTem: batch_tem_x, xSta: batch_sta_x, y: batch_y})
            # Calculate batch loss
            lossTrain = sess.run(cost, feed_dict={xTem: batch_tem_x, xSta: batch_sta_x, y: batch_y})
            accTest = sess.run(accuracy, feed_dict={xTem: test_tem_data, xSta: test_sta_data, y: test_label})
            lossTest = sess.run(cost, feed_dict={xTem: test_tem_data, xSta: test_sta_data, y: test_label})
            test_auc = sess.run(auc, feed_dict={xTem: test_tem_data, xSta: test_sta_data, y: test_label})
            train_auc = sess.run(auc, feed_dict={xTem: batch_tem_x, xSta: batch_sta_x, y: batch_y})
            # Calculate batch accuracy
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(lossTrain) + ", Test Loss= " + "{:.6f}".format(lossTest) +", Training Accuracy= " + \
                          "{:.5f}".format(accTrain),  ", Test Accuracy= " + \
                          "{:.5f}".format(accTest),", Train AUC= " + "{:.6f}".format(train_auc[1]),", Test AUC= " + "{:.6f}".format(test_auc[1]))
        step += 1
    print("Optimization Finished!")
print('single iteration finish')

