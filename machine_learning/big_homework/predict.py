import torch
import VGG16
import torchvision
import torchvision.transforms as transforms

# 选择训练环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据归一化
transform = transforms.Compose(
        [
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
)

# cifar10数据标签
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

checkpoint = torch.load('./checkpoint/cifar10_epoch_5.ckpt', map_location='cpu')
net = VGG16.VGG('VGG16').to(device)
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

if __name__ == '__main__':
    # 载入数据
    testset = torchvision.datasets.CIFAR10(
        root='./Cifar10',
        train=False,
        transform=transform,
        download=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        num_workers=2,
        batch_size=4,
        shuffle=True
    )

    ##################
    # 计算测试集预测效果
    ##################
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (cifar10_classes[i], 100 * class_correct[i] / class_total[i]))