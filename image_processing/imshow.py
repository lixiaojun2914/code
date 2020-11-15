import cv2
import numpy as np

# 读取图片
image = cv2.imread("cherry.png")
R = image[:, :, 2]
G = image[:, :, 1]
B = image[:, :, 0]

# RGB2GRAY
img = (0.2989 * R + 0.5870 * G + 0.1140 * B)
# img = (R + G + B) / 3.0

def randCent(k):
    return 255 * np.random.rand(k)

def kMeans(dataSet, k):
    (m, n) = dataSet.shape
    clusterAssment = np.zeros((m, n, 2))
    centroids = randCent(k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            for j in range(n):
                minDist = np.inf
                minIndex = -1
                for cent in range(k):
                    dist = np.abs(dataSet[i][j] - centroids[cent])
                    if dist < minDist:
                        minDist = dist
                        minIndex = cent
                if clusterAssment[i][j][0] != minIndex:
                    clusterChanged = True
                clusterAssment[i][j] = [minIndex, minDist]
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment[:, :, 0] == cent]
            centroids[cent] = np.mean(ptsInClust)
    return centroids, clusterAssment

def result(dataSet, k):
    (m, n) = dataSet.shape
    myCentroids, clusterAssing = kMeans(dataSet, k)
    ans = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # ans[i, j] = myCentroids[int(clusterAssing[i, j, 0])]
            if clusterAssing[i, j, 0] == 0:
                ans[i, j] = 0
            else:
                ans[i, j] = 255
    return ans


ans = result(img, 2)
cv2.imshow("img", img/255)
cv2.imshow("ans", ans/255)

key = cv2.waitKey()
if key == 27:
    cv2.destroyWindow("img")
