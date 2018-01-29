#!/usr/env python
# -*- coding=utf-8 -*-

import numpy as np
import  matplotlib.pyplot as plt

#traing examples
m = 201

#set data
x = np.random.randint(100000, size=m)
noise = np.random.randint(-5000, 5000, size=m)
y = x + noise

# divide all examples into training data and testing data
Xt = x[:150]
Yt = y[:150]
Xte  = x[151:]
Yte  = y[151:]

Xt = Xt.reshape(150, 1)
Yt = Yt.reshape(150, 1)
Xte = Xte.reshape(50, 1)
Yte = Yte.reshape(50, 1)

# normal equation
Xt_T = np.transpose([Xt])
Xt_T = Xt_T.reshape(1, 150)
Xt_T_dot_Xt = Xt_T.dot(Xt)
Xt_T_dot_Xt_inv = np.linalg.inv(Xt_T_dot_Xt)
Xt_T_dot_Xt_inv_dot_Xt_T = Xt_T_dot_Xt_inv.dot(Xt_T)

theta = Xt_T_dot_Xt_inv_dot_Xt_T.dot(Yt)
theta1 = theta[0][0]
theta0 = 1100  # self-define

#model
Y_pred = Xte * theta1 + theta0


#plot data
fig = plt.figure()
plot1 = fig.add_subplot(1,1,1)
plt.xlim(0, 100000)
plt.ylim(0, 100000)
plot1.scatter(Xt, Yt)
plt.ion()
plt.show()
lines = plot1.plot(Xte, Y_pred, 'r-', lw=3)
 

#test
cost = np.sum(abs(Y_pred - Yte))
print("test cost: ", cost)

plt.pause(5)
