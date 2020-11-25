import numpy as np
import cv2
import matplotlib.pyplot as plt


# sobel算子计算
def sobel(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Y方向
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] ** 2 + new_imageY[i + 1, j + 1] ** 2) ** 0.5
    return np.uint8(new_imageX), np.uint8(new_imageY), np.uint8(new_image)


# 图像cell分割
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


# cell直方图计算
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = grad_cell[i, j].flatten()
            ang_list = ang_cell[i, j].flatten()
            left = np.int8(ang_list / 20.0)
            right = left + 1
            right[right >= 8] = 0
            left_rit = (ang_list - 20 * left) / 20.0
            right_rit = 1.0 - left_rit
            binn[left] += left_rit * grad_list
            binn[right] += right_rit * grad_list
            bins[i, j] = binn
    return bins


# 提取hog特征
def hog(img, cell_x, cell_y, cell_w):
    # 计算梯度
    gx, gy, grad = sobel(img)
    # 根据梯度值计算角度
    ang = np.arctan2(gx, gy)
    # 将角度同意到0-180度
    ang[ang < 0] = np.pi + ang[ang < 0]
    ang *= (180.0 / np.pi)
    ang[ang >= 180] -= 180
    # 对梯度矩阵和方向矩阵进行cell的划分
    grad_cell = div(grad, cell_x, cell_y, cell_w)
    ang_cell = div(ang, cell_x, cell_y, cell_w)
    # 计算bins
    bins = get_bins(grad_cell, ang_cell)
    # 计算特征向量
    feature = []
    # 四个cell为一个block，计算每个block的特征，并加入hog特征矩阵
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = [bins[i, j], bins[i + 1, j], bins[i, j + 1], bins[i + 1, j + 1]]
            # 归一化
            tmp -= np.mean(tmp)
            feature.append(np.array(tmp).flatten())
    return np.array(feature).flatten()


# 读取灰度图像
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
# 计算Hog特征
cell_w = 8
cell_x = int(img.shape[0] / cell_w)
cell_y = int(img.shape[1] / cell_w)
feature = hog(img, cell_x, cell_y, cell_w)
print(feature)
