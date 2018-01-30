import numpy as np
import 	matplotlib.pyplot as plt

x = np.array([ 2, 3  , 4, 5,  6,  7,  8,    9, 10, 11,  12,  13, 14]).astype(float)
y = np.array([ 4, 4.5, 6, 9, 11, 12, 12.3, 12, 11,  9,   6, 4.5,  4]).astype(float)


plt.xlim(0, 16)
plt.ylim(0, 15)
plt.plot(x, y)
plt.plot(x[3:6], y[3:6], 'r', lw = 5)
plt.plot(x[5], y[5], 'k')
plt.plot(np.linspace(6, 6, 5), np.linspace(0, y[4], 5), 'k--')
plt.show()
