""" Script to produce the loss plot """

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('loss.csv', delimiter=',', names=['l1', 'l2', 'l3'])
y1 = np.array([x[0] for x in data])
x1 = np.array([x for x in range(len(y1))])
y2 = np.array([x[1] for x in data])
x2 = np.array([x for x in range(len(y2))])
y3 = np.array([x[2] for x in data])
x3 = np.array([x for x in range(len(y3))])

fig = plt.figure()

ax1 = fig.add_subplot()
ax1.plot(x1, y1, c='r', label='Epoch 1')

ax2 = fig.add_subplot()
ax2.plot(x2, y2, c='g', label='Epoch 2')

ax3 = fig.add_subplot()
ax3.plot(x3, y3, c='b', label='Epoch 3')

plt.xlabel('Epoch Iterations')
plt.ylabel('Loss (logit)')
plt.title('Loss per Epoch Iterations')
plt.legend()
plt.savefig('loss.png')
