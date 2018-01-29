#!/usr/env python
# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf
from pylab import *

#set data
x_data = np.array([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
                   1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
                   1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637,
                   1839, 1000, 2040, 3137, 1811, 1437, 1239, 2132, 4215, 2162,
                   1664, 2238, 2567, 1200,  852, 1852, 1203])

y_data = np.array([399900,329900,369000,232000,539900,299900,314900,198999,212000,242500,
                   239999,347000,329999,699900,259900,449900,299900,199900,499998,599000,
                   252900,255000,242900,259900,573900,249900,464500,469000,475000,299900,
                   349900,169900,314900,579900,285900,249900,229900,345000,549000,287000,
                   368500,329900,314000,299000,179900,299900,239500])


x_data = np.reshape(x_data, (47,1))
y_data = np.reshape(y_data, (47,1))

X_train = x_data[:]
Y_train = y_data[:]
X_test = x_data[:]
Y_test = y_data[:]


# set up a neural layer
def add_layer(feature1, in_size, out_size, activation_function=None):
    theta0 = tf.Variable(tf.zeros([in_size, out_size]))
    theta1 = tf.Variable(tf.zeros([1, out_size]))
    y_pred = tf.matmul(feature1, theta1) + theta0
    if activation_function is None:
        outputs = y_pred
    else:
        outputs = activation_function(y_pred)
    return outputs


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])


prediction = add_layer(x, 1, 1, activation_function=None)
loss = tf.reduce_sum(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.00000000001).minimize(loss)
init = tf.global_variables_initializer()

# create a session 
sess = tf.Session()
sess.run(init)


# plot the trainging data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.xlim(500, 5000)
plt.ylim(0, 1000)
ax.scatter(x_data, y_data/1000)
plt.ion()
plt.show()


# training
for i in range(3000):
    sess.run(train_step, feed_dict={x: x_data, y: y_data})
    if i % 500 == 0 :
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        pred = sess.run(prediction, feed_dict={x: x_data})
        lines = ax.plot(x_data, pred/1000, 'g-', lw=2)
        plt.pause(0.5)

# evaluate
cost = sess.run( tf.reduce_sum(tf.abs(pred - Y_test))/47)
print("cost: ", cost)
plt.pause(5)
