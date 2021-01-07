import numpy as np
import operator


class Knn:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train

    def predict(self, k, dis, X_test):
        # assert dis == 'E' or dis == 'M', 'dis must E or M'
        num_test = X_test.shape[0]  # 测试样本的数量
        labellist = []
        # 使用欧拉公式作为距离度量
        if dis == 'E':
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))) ** 2), axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for j in topK:
                    classCount[self.ytr[j]] = classCount.get(self.ytr[j], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)

        # 使用曼哈顿公式作为距离度量
        if dis == 'M':
            for i in range(num_test):
                # 按照列的方向相加，其实就是行相加
                distances = np.sum(np.abs(self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))), axis=1)
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for j in topK:
                    classCount[self.ytr[j]] = classCount.get(self.ytr[j], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)
