#!/usr/env python
# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import  matplotlib.pyplot as plt

m = 200

#set data
x = np.random.randint(100000, size=m)
noise = np.linspace(-5000, 5000, m)
y = x + noise


X_train = x[:]
Y_train = y[:]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

sgd = optimizers.SGD(lr=0.0000000001)
model.compile(loss='mse', optimizer=sgd)

#plot data
fig = plt.figure()
plot1 = fig.add_subplot(1,1,1)
plt.xlim(0, 100000)
plt.ylim(0, 100000)
plot1.scatter(x, y)
plt.ion()
plt.show()

print('Training.....')
for i in range(5000):
  if i % 200 == 0 :
    try:
      plot1.lines.remove(lines[0])
    except Exception:
      pass
    cost = model.train_on_batch(X_train, Y_train)
    print('train cose: ', cost)
    Y_pred = model.predict(X_train)
    lines = plot1.plot(X_train, Y_pred, 'r-', lw=3)
    plt.pause(0.5)
 
