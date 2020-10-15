import numpy as np
import matplotlib.pyplot as plt

## 分离超平面
## y = 3x-2

x = 10 * np.random.rand(50, 2)
x1 = []
x2 = []

y = []
for i in range(50):
    if x[i][1] > (3*x[i][0]-2):
        y.append(1)
    else:
        y.append(-1)

for i in range(50):
    lam = 2 * np.random.rand() - 1
    if x[i][1] > (3*x[i][0]-2+lam):
        x1.append(x[i])
    else:
        x2.append(x[i])

x1 = np.array(x1)
x2 = np.array(x2)

lr = 0.01
w = np.zeros((2, 1))
b = 0.0
pre_loss = 0

for i in range(1000):
    y_ = np.sign(np.matmul(w, x) + b)
    ##y_(wx+b)>=0
    loss = np.sum(np.abs(y+y_)/2)
    if loss > pre_loss:
        pre_loss = loss
        w -= lr * np.dot(y_, x)
        b -= lr * y_

plt.axis([0, 10, 0, 10])
plt.scatter(x1[:,0], x1[:,1], c='red')
plt.scatter(x2[:,0], x2[:,1], c='blue')
xx = np.linspace(0, 10, 100)
yy = 3*xx-2
plt.plot(xx, yy)
