# -*- coding=utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf

#set data
X = np.array([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
              1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
              1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637,
              1839, 1000, 2040, 3137, 1811, 1437, 1239, 2132, 4215, 2162,
              1664, 2238, 2567, 1200,  852, 1852, 1203])

Y = np.array([399900,329900,369000,232000,539900,299900,314900,198999,212000,242500,
              239999,347000,329999,699900,259900,449900,299900,199900,499998,599000,
              252900,255000,242900,259900,573900,249900,464500,469000,475000,299900,
              349900,169900,314900,579900,285900,249900,229900,345000,549000,287000,
              368500,329900,314000,299000,179900,299900,239500])

# 因為數據太少，所以訓練集 = 測試集
X_train, Y_train = X[:], Y[:]
X_test, Y_test = X[:], Y[:]     


features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
# 定義模型，此例使用線性迴歸模型
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# 自訂模型
def model(features, labels, mode):
  # 建立線性迴歸模型
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # loss function 的 sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # 訓練的 sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.00000000001)
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  # 使用 ModelFnOps 將我們建立的 subgraphs 包裝好
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":X_train}, Y_train, batch_size = 47, num_epochs = 10)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":X_test}, Y_test, batch_size = 47, num_epochs = 10)

# 進行模型的訓練
estimator.fit(input_fn = input_fn, steps = 100000)

# 驗證模型
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
