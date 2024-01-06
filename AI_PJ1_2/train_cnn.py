import torchvision
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from cnn_model import ConvNet
import torch.optim as optim

EPOCHS = 10
BATCH_SIZE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

data_path = './train/'
test_data_path = './test/'

dataSet = torchvision.datasets.ImageFolder(data_path, transform=transform)
test_dataSet = torchvision.datasets.ImageFolder(test_data_path, transform=transform)

total_samples = len(dataSet)

train_size = int(0.8 * total_samples)
val_size = total_samples - train_size

train_dataset, val_dataset = random_split(dataSet, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataSet, batch_size=BATCH_SIZE, shuffle=False)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):  ## 使用enumerate访问可遍历的数组对象
        data, label = data.to(device), label.to(device)
        output = model(data)
        train_loss = F.nll_loss(output, label)  # 计算误差
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == label.data)  # 计算准确度
        optimizer.zero_grad()  # 清空上一次的梯度
        total += label.size(0)  # 更新训练样本数
        train_loss.backward()  # 误差反向传递
        optimizer.step()  # 优化器参数更新
        if (batch_idx + 1) % 30 == 0:  # 每训练30组(30*50个)数据输出1次结果
            print('Train epoch{}/{}: [{}/{} ({:.0f}%)]\ttrain_loss: {:.6f},train accuracy:（{:.0f}%）'.format(
                epoch, EPOCHS, total, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                train_loss.item(),  # train_loss
                100 * correct / total))  # train accuracy


# 测试过程
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = torch.max(output.data, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    mode = "train"
    if mode == "train":
        model = ConvNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters())
        for epoch in range(1, EPOCHS + 1):
            train(model, DEVICE, train_loader, optimizer, epoch)
            torch.save(model.state_dict(), 'my_model.pth')
        model2 = ConvNet().to(DEVICE)
        model2.load_state_dict(torch.load('my_model.pth'))  # 加载已保存的状态字典
        test(model2, DEVICE, val_loader)
    elif mode == "test":
        model_test = ConvNet().to(DEVICE)
        model_test.load_state_dict(torch.load('my_model.pth'))  # 加载已保存的状态字典
        test(model_test, DEVICE, test_loader)
