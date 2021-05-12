import numpy as np
from collections import OrderedDict


def softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)


def cross_entropy_error(p, y):
    delta = 1e-7
    batch_size = p.shape[0]
    return -np.sum(y * np.log(p + delta)) / batch_size


def one_hot(y, class_num):
    code = np.eye(class_num)
    return code[y]


class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.maximum(0, x)
        out = self.x
        return out

    def backward(self, dout):
        dx = dout
        dx[self.x <= 0] = 0
        return dx


class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.p = None
        self.y = None

    def forward(self, x, y):
        self.y = y
        self.p = softmax(x)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.p - self.y) / batch_size
        return dx


class Net:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['w3'] = weight_init_std * np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['layer1'] = Linear(self.params['w1'], self.params['b1'])
        self.layers['relu1'] = Relu()
        self.layers['layer2'] = Linear(self.params['w2'], self.params['b2'])
        self.layers['relu2'] = Relu()
        self.layers['layer3'] = Linear(self.params['w3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        p = self.predict(x)
        return self.last_layer.forward(p, y)

    def accuracy(self, x, y):
        p = self.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == p) / float(x.shape[0])
        return accuracy

    def gradient(self, x, y):
        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['w1'], grads['b1'] = self.layers['layer1'].dw, self.layers['layer1'].db
        grads['w2'], grads['b2'] = self.layers['layer2'].dw, self.layers['layer2'].db
        grads['w3'], grads['b3'] = self.layers['layer3'].dw, self.layers['layer3'].db
        return grads


# data_prepare
path = './dataset/mnist.npz'
with np.load(path)as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_train, y_test = one_hot(y_train, 10), one_hot(y_test, 10)

# train
train_size = x_train.shape[0]
iters_num = 600
learning_rate = 0.01
epoch = 5
batch_size = 100

net = Net(784, 50, 50, 10)
for i in range(epoch):
    print(f'current epoch is :{i+1}')
    for num in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = net.gradient(x_batch, y_batch)

        for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
            net.params[key] -= learning_rate * grad[key]

        loss = net.loss(x_batch, y_batch)
        if num % 100 == 0:
            print(loss)

    print(f'accuracy: {net.accuracy(x_test, y_test) * 100} %')
