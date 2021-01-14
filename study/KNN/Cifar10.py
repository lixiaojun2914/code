import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import KNN2

batch_size = 100

# MNIST dataset
train_dataset = dsets.CIFAR10(
    root='./cf/pymnist',
    train=True,
    transform=None,
    download=True
)

test_dataset = dsets.CIFAR10(
    root='./cf/pymnist',
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


X_train = train_loader.dataset.data[:200]
mean_image = getXmean(X_train)
X_train = centralized(X_train, mean_image)
y_train = np.array(train_loader.dataset.targets[:200])
X_test = test_loader.dataset.data[:40]
X_test = centralized(X_test, mean_image)
y_test = np.array(test_loader.dataset.targets[:40])

# k折交叉验证
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20]
num_training = X_train.shape[0]
X_train_folds = []
y_train_folds = []
indices = np.array_split(np.arange(num_training), indices_or_sections=num_folds)
for i in indices:
    X_train_folds.append(X_train[i])
    y_train_folds.append(y_train[i])

k_to_accuracies = {}
for k in k_choices:
    acc = []
    for i in range(num_folds):
        x = X_train_folds[0:i] + X_train_folds[i + 1:]
        x = np.concatenate(x, axis=0)

        y = y_train_folds[0:i] + y_train_folds[i + 1:]
        y = np.concatenate(y)

        test_x = X_train_folds[i]
        test_y = y_train_folds[i]

        classifier = KNN2.Knn()
        classifier.fit(x, y)

        y_pred = classifier.predict(k, 'M', test_x)
        accuracy = np.mean(y_pred == test_y)
        acc.append(accuracy)

    k_to_accuracies[k] = acc

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
