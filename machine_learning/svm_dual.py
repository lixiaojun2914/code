import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
import time

start = time.time()

# 数据生成
# 分离超平面
# y = 3x-2


# 网络输入
# N：坐标个数
# D_in：输入位数
# D_out: 输出维度

N, D_in, D_out = 1000, 2, 1
epochs = 1000

# 随机生成数据
np.random.seed(6)
x = 10 * np.random.rand(N + 200, D_in)

# 将坐标分类
y = []
for i in range(1200):
    if x[i][1] > (3 * x[i][0] - 2):
        y.append(1.0)
    else:
        y.append(-1.0)

# 加入噪声
zz = 0
for i in range(1200):
    lam = zz * np.random.rand() - zz
    x[i][0] += lam

y = np.array(y)

# 1000训练，200测试
x_train = x[:N]
y_train = y[:N]

x_test = x[N:]
y_test = y[N:]

Q = cvx.matrix(np.zeros((N, N)))
p = cvx.matrix([-1.0] * N)
G = cvx.matrix(-np.eye(N))
h = cvx.matrix([0.0] * N)
A = cvx.matrix(y_train, (1, 1000))
b = cvx.matrix(0.0)

for i in range(N):
    for j in range(N):
        Q[i, j] = y_train[i] * y_train[j] * np.matmul(x_train[i], x_train[j].T)

sol = cvx.solvers.qp(Q, p, G, h, A, b)

a = np.array(sol['x'])
w = np.matmul(x_train.T, (a * np.array([y_train]).T))

lens = 0
for i in range(N):
    if a[i] > 1e-5:
        b += y_train[i]
        b -= np.matmul(x_train[i], w)
        lens += 1
b /= lens
w = np.append(w, [b])

# 打印分割超平面
print("----------------------------------")
print("y = %d * x + %d" % (-w[0] / w[1], -w[2] / w[1]))

# 打印运行时间
end = time.time()
print("----------------------------------")
print("time: ", end - start)

# 预测
y_pred = np.sign(np.matmul(x_test, w[:D_in]) + w[D_in:])
acc = np.mean(y_pred.flatten() == y_test)
print("----------------------------------")
print("acc: ", acc * 100, "%")

# 绘图
xx = np.linspace(0, 10, 100)
yy = -(w[0] * xx + w[2]) / w[1]

# 训练集点分类
x1 = []
x2 = []
for i in range(1000):
    if y_train[i] == 1:
        x1.append(x[i])
    else:
        x2.append(x[i])
x1 = np.array(x1)
x2 = np.array(x2)

# 测试集点分类
r1 = []
r2 = []
for i in range(200):
    if y_test[i] == 1:
        r1.append(x_test[i])
    else:
        r2.append(x_test[i])
r1 = np.array(r1)
r2 = np.array(r2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# 训练集绘制
ax1.set_title("train set")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.scatter(x1[:, 0], x1[:, 1], c='red')
ax1.scatter(x2[:, 0], x2[:, 1], c='blue')
ax1.plot(xx, yy, lw=2)

# 测试集绘制
ax2.set_title("test set")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.scatter(r1[:, 0], r1[:, 1], c='red')
ax2.scatter(r2[:, 0], r2[:, 1], c='blue')
ax2.plot(xx, yy, 'g-', lw=2)

plt.show()
