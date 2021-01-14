import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import os
import time
import torch.optim as optim
import matplotlib.pyplot as plt

####################
# VGG16网络结构定义
####################

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG(nn.Module):
    def __init__(self, net_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[net_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
    # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # kaiming分布初始化
                # pytorch默认使用kaiming分布初始化卷积层
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    ####################
    # Cifar10数据加载
    ####################

    # 选择训练环境
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据归一化
    transform = transforms.Compose(
        [
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 载入数据
    trainset = torchvision.datasets.CIFAR10(
        root='./Cifar10',
        train=True,
        transform=transform,
        download=True
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        num_workers=2,
        batch_size=4,
        shuffle=True
    )

    net = VGG('VGG16').to(device)

    ####################
    # 定义损失函数和优化方法
    ####################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ####################
    # 网络训练
    ####################
    start_time = time.perf_counter()

    for epoch in range(5):
        train_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            # 初始化
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # 优化过程
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            # 查看网络训练状态
            train_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, train_loss / 2000))
                train_loss = 0.0

        # 保存训练模型
        print('Saving epoch %d model ...' % (epoch+1))
        state = {
            'net': net.state_dict(),
            'epoch': epoch + 1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/cifar10_epoch_%d.ckpt' % (epoch + 1))

    print('Finish Training')
    end_time = time.perf_counter()
    print('Running time: %.3f Seconds' % (end_time - start_time))
