from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7* 7 * 32, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 12),
        )

    def forward(self, x):
        # print(x.shape)
        covn1_out = self.layer1(x)
        # print(covn1_out.shape)
        covn2_out = self.layer2(covn1_out)
        # print(covn2_out.shape)
        out = covn2_out.view(x.size(0), -1)#将卷积层的输出 conv2_out 从一个三维张量转换为一个二维张量，
        # 其中每一行代表一个批次中的一个样本，每一列代表一个特征值。
        # 这是为了将卷积层提取的特征图转换为全连接层能够接受的形式。
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out