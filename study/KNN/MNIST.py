import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import KNN

batch_size = 100

# MNIST dataset
train_dataset = dsets.MNIST(
    root='./ml/pymnist',
    train=True,
    transform=None,
    download=True
)

test_dataset = dsets.MNIST(
    root='./ml/pymnist',
    train=False,
    transform=None,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

def getXmean(X_train):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    mean_image = np.mean(X_train, axis=0)
    return mean_image

def centralized(X_test, mean_image):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_test = X_test.astype(np.float)
    X_test -= mean_image
    return X_test


X_train = train_loader.dataset.data.numpy()
mean_image = getXmean(X_train)
X_train = centralized(X_train, mean_image)
y_train = train_loader.dataset.targets.numpy()
X_test = test_loader.dataset.data[:1000].numpy()
X_test = centralized(X_test, mean_image)
y_test = test_loader.dataset.targets[:1000].numpy()
num_test = y_test.shape[0]
y_test_pred = KNN.kNN_classify(5, 'M', X_train, y_train, X_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
