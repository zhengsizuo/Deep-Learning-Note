"""
RNN classifier for MNIST dataset, cells are basic type or LSTM-type.
Author: zhs
Date: Jan 16, 2019
"""

import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")
# 训练参数
n_epoches = 100
batch_size = 150
Learning_rate = 0.001
# 网络参数，把28x28的图片数据拆成28行的时序数据喂进RNN
n_inputs = 28
n_steps = 28
n_hiddens = 150
n_outputs = 10  # 10分类


# 输入tensors
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# 构建RNN结构
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hiddens, state_is_tuple=True)
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_hiddens)
# basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_hiddens)  # 另一种创建基本单元的方式
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# 前向传播，定义损失函数、优化器
logits = tf.layers.dense(states[-1], n_outputs)  # 与states tensor连接的全连接层，LSTM时为states[-1]，即h张量
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate)
train_op = optimizer.minimize(loss)

prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))  # cast函数将tensor转换为指定类型

# 从MNIST中读取数据
X_test = mnist.test.images.reshape([-1, n_steps, n_inputs])
y_test = mnist.test.labels

# 训练阶段
init = tf.global_variables_initializer()
loss_list = []
accuracy_list = []

with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size  # 整除返回整数部分
    # print("Batch_number: {}".format(n_batches))
    for epoch in range(n_epoches):
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape([-1, n_steps, n_inputs])
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        loss_train = loss.eval(feed_dict={X: X_batch, y: y_batch})
        loss_list.append(loss_train)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        accuracy_list.append(acc_test)
        print(epoch, "Train accuracy: {:.3f}".format(acc_train), "Test accuracy: {:.3f}".format(acc_test))

# 导出损失和准确率，方便绘图
loss_readout = pd.DataFrame(loss_list)
loss_readout.to_csv('csv/RNN_LSTM_loss.csv')
acc_readout = pd.DataFrame(accuracy_list)
acc_readout.to_csv('csv/RNN_LSTM_accuracy.csv')
