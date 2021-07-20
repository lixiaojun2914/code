import pickle
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
path = './dataset/mnist.npz'
with np.load(path)as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# 随机初始化聚类中心
def randCent(k):
    return 255 * np.random.rand(k, 28, 28)


def kMeans(dataSet, k):
    n = dataSet.shape[0]
    # 记录每个点到聚类中心的类别和距离
    clusterAssment = np.zeros((n, 2))
    # 聚类中心
    centroids = randCent(k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(n):
            minDist = np.inf
            minIndex = -1
            for cent in range(k):
                dist = np.sqrt(np.sum(np.power(dataSet[i] - centroids[cent], 2)))
                if dist < minDist:
                    minDist = dist
                    minIndex = cent
            # 如果没有可更新的， 就退出循环
            if clusterAssment[i][0] != minIndex:
                clusterChanged = True
            clusterAssment[i] = [minIndex, minDist]
        # 根据距离重新计算聚类中心
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment[:, 0] == cent]
            centroids[cent] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment




# myCentroids, clusterAssing = kMeans(x_train, 10)
# with open('./cent.save', 'wb') as f:
#     pickle.dump(myCentroids, f)
# with open('./cluster.save', 'wb') as f:
#     pickle.dump(clusterAssing, f)


with open('./cent.save', 'rb') as f:
    myCentroids = pickle.load(f)
with open('./cluster.save', 'rb') as f:
    clusterAssing = pickle.load(f)
print('done！')

a = np.array([np.bincount(y_train[clusterAssing[:, 0] == i]).argmax() if len(y_train[clusterAssing[:, 0] == i]) else -1 for i in range(10)])
print(a)
# a = [np.bincount(y_train[clusterAssing[:, 0] == i]).argmax() for i in range(10)]
# print(a)

acc = 0
for i in range(10):
    acc += np.sum((y_train[clusterAssing[:, 0] == i]) == a[i])
acc /= 60000
print(f'accuracy: {acc * 100:.2f}%')

plt.figure()
plt.suptitle('Cluster Center Visualization', y=0.9)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.title(a[i], y=-0.5)
    plt.imshow(myCentroids[i])
plt.show()
