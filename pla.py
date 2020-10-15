import numpy as np
import matplotlib.pyplot as plt

## 分离超平面
## y = 3x-2

N, D_in, D_out = 1000, 2, 1


x = 10 * np.random.rand(N, D_in)
x1 = []
x2 = []
y = []
for i in range(N):
    if x[i][1] > (3*x[i][0]-2):
        y.append(1)
    else:
        y.append(-1)

for i in range(N):
    lam = 5 * np.random.rand() - 5
    if x[i][1] > (3*x[i][0]-2+lam):
        x1.append(x[i])
    else:
        x2.append(x[i])

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)

lr = 0.0001
w = np.random.randn(D_in, D_out)
b = 0.0

for i in range(1000):
    y_ = np.sign(np.matmul(x, w) + b)
    for i in range(N):
        if y_[i][0] != y[i]:
            w += lr * y[i] * np.array([x[i]]).T
            b += lr * y[i]
    loss = np.sum(y!=y_.flatten())
    print(loss)

plt.subplot(1, 2, 1)
plt.axis([0, 10, 0, 10])
plt.scatter(x1[:,0], x1[:,1], c='red')
plt.scatter(x2[:,0], x2[:,1], c='blue')
xx = np.linspace(0, 10, 100)
yy = 3*xx-2
plt.plot(xx, yy)

r1 = []
r2 = []
res = np.sign(np.matmul(x, w) + b)
for i in range(N):
    if res[i][0] == 1:
        r1.append(x[i])
    else:
        r2.append(x[i])
r1 = np.array(r1)
r2 = np.array(r2)

plt.subplot(1, 2, 2)
plt.axis([0, 10, 0, 10])
plt.scatter(r1[:,0], r1[:,1], c='red')
plt.scatter(r2[:,0], r2[:,1], c='blue')

plt.show()