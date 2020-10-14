import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

w = 0.0
b = 0.0




plt.axis([0, 10, 0, 10])
plt.scatter(x1[:,0], x1[:,1], c='red')
plt.scatter(x2[:,0], x2[:,1], c='blue')
xx = np.linspace(0, 10, 100)
yy = 3*xx-2
plt.plot(xx, yy)
plt.show()