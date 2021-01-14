import numpy as np
import operator


def kNN_classify(k, dis, X_train, x_train, Y_test):
    assert dis == 'E' or dis == 'M', 'dis must E or M，E代表欧拉距离，M代表曼哈顿距离'
    num_test = Y_test.shape[0]  # 测试样本的数量
    labellist = []
    '''
    使用欧拉公式作为距离度量
    '''
    if dis == 'E':
        for i in range(num_test):
            # 实现欧拉距离公式
            distances = np.sqrt(np.sum(((X_train - np.tile(Y_test[i], (X_train.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)  # 距离由小到大进行排序，并返回index值
            topK = nearest_k[:k]  # 选取前k个距离
            classCount = {}
            for j in topK:  # 统计每个类别的个数
                classCount[x_train[j]] = classCount.get(Y_test[j], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

    if dis == 'M':
        for i in range(num_test):
            # 实现曼哈顿距离公式
            distances = np.sum(np.abs(X_train - np.tile(Y_test[i], (X_train.shape[0], 1))), axis=1)
            nearest_k = np.argsort(distances)  # 距离由小到大进行排序，并返回index值
            topK = nearest_k[:k]  # 选取前k个距离
            classCount = {}
            for j in topK:  # 统计每个类别的个数
                classCount[x_train[j]] = classCount.get(x_train[j], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
