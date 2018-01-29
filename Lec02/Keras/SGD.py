# -*- coding=utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt 
import numpy as np

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

# set model
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
sgd = optimizers.SGD(lr = 0.000000001)
model.compile(loss='mse', optimizer= sgd)

# training
print('Training -----------')
for step in range(500):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=47)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()	
