import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 数据生成
# 分离超平面
# y = 3x-2


# 网络输入
# N：坐标个数
# D_in：输入位数
# D_out: 输出维度

N, D_in, D_out = 1000, 3, 1
epochs = 1000

# 随机生成数据
np.random.seed(6)
x = 10 * np.random.rand(N + 200, D_in - 1)

# 添加一列1，将b并入矩阵
x = np.insert(x, D_in - 1, 1, axis=1)

# 将坐标分类
y = []
for i in range(1200):
    if x[i][1] > (3 * x[i][0] - 2):
        y.append(1)
    else:
        y.append(-1)

# 加入噪声
zz = 3
for i in range(1200):
    lam = zz * np.random.rand() - zz
    x[i][0] += lam

y = np.array(y)

# 1000训练，200测试
x_train = x[:N]
y_train = y[:N]

x_test = x[N:]
y_test = y[N:]

lr = 0.0001
w = np.random.randn(D_in, D_out)

# 绘图历史信息
w_draw = []
loss_draw = []
acc_draw = []

# 深拷贝
w_best = w.copy()
for epoch in range(epochs):
    y_ = np.sign(np.matmul(x_train, w))
    y_best = np.sign(np.matmul(x_train, w_best))
    for i in range(N):
        if y_[i][0] != y_train[i]:
            w += lr * y_train[i] * np.array([x_train[i]]).T

    loss = np.sum(y_train != y_.flatten())
    loss_best = np.sum(y_train != y_best.flatten())
    if (loss < loss_best):
        w_best = w.copy()
        loss_draw.append(loss)
    else:
        loss_draw.append(loss_best)

    test = np.sign(np.matmul(x_test, w_best))
    acc = np.sum(test.flatten() == y_test) / len(test) * 100
    acc_draw.append(acc)

    if epoch % 100 == 0 or epoch + 1 == epochs:
        print("epoch: ", epoch)
        w_draw.append(w_best.copy())
        print("----------------------------------------")
        print("Loss: ", loss, ", Acc: ", acc, "%")
        print("line: y = ", -w_best[0] / w_best[1], "* x + ", -w_best[2] / w_best[1], "\n\n")
w = w_best.copy()

# 绘图
xx = np.linspace(0, 10, 100)
yy = -(w[0] * xx + w[2]) / w[1]

# 训练集点分类
x1 = []
x2 = []
for i in range(1000):
    if y[i] == 1:
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))


# 训练集绘制
def init():
    line = ax1.plot(xx, -(w_draw[0][0] * xx + w_draw[0][2]) / w_draw[0][1], lw=2)
    ax1.scatter(x1[:, 0], x1[:, 1], c='red')
    ax1.scatter(x2[:, 0], x2[:, 1], c='blue')
    return line


def animate(i):
    line = ax1.plot(xx, -(w_draw[i][0] * xx + w_draw[i][2]) / w_draw[i][1], lw=2)
    ax1.scatter(x1[:, 0], x1[:, 1], c='red')
    ax1.scatter(x2[:, 0], x2[:, 1], c='blue')
    return line


ax1.set_title("train set")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ani = animation.FuncAnimation(fig=fig, func=animate, \
                              frames=len(w_draw), init_func=init, interval=500, \
                              blit=True, repeat=False)

# 测试集绘制
ax2.set_title("test set")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.scatter(r1[:, 0], r1[:, 1], c='red')
ax2.scatter(r2[:, 0], r2[:, 1], c='blue')
ax2.plot(xx, yy, 'g-', lw=2)

lax = np.linspace(0, epochs, 1000)
# Loss曲线绘制
ax3.set_title("Loss")
ax3.plot(lax, loss_draw)

# acc曲线绘制
ax4.set_title("Acc")
ax4.plot(lax, acc_draw)

plt.show()
