import torch.nn as nn


# 建立神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # 第一个卷积层，接收5个通道的输入，输出32个通道
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 对应的Batch Normalization层
        self.relu1 = nn.ReLU()  # 对应的ReLU激活函数

        # 第二个卷积层，接收32个通道的输入，输出64个通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 对应的Batch Normalization层
        self.relu2 = nn.ReLU()  # 对应的ReLU激活函数

        # 第三个卷积层，接收64个通道的输入，输出128个通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 对应的Batch Normalization层
        self.relu3 = nn.ReLU()  # 对应的ReLU激活函数

        # 第四个卷积层，接收128个通道的输入，输出256个通道
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # 对应的Batch Normalization层
        self.relu4 = nn.ReLU()  # 对应的ReLU激活函数

        # 第五个卷积层，接收256个通道的输入，输出128个通道
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)  # 对应的Batch Normalization层
        self.relu5 = nn.ReLU()  # 对应的ReLU激活函数

        # 策略头，用于预测下一步的走法
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),  # 1x1卷积层
            nn.BatchNorm2d(2),  # 对应的Batch Normalization层
            nn.ReLU(),  # ReLU激活函数
            nn.Flatten(),  # 展平操作
            nn.Linear(2 * 15 * 15, 15 * 15)  # 全连接层
        )

        # 价值头，用于评估当前局面的价值
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),  # 1x1卷积层
            nn.BatchNorm2d(1),  # 对应的Batch Normalization层
            nn.ReLU(),  # ReLU激活函数
            nn.Flatten(),  # 展平操作
            nn.Linear(15 * 15, 64),  # 全连接层，将15x15的输出映射到64个神经元
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(64, 1),  # 全连接层，将64个神经元映射到1个输出
            nn.Tanh()  # Tanh激活函数，将输出值限制在-1到1之间
        )

    def forward(self, x):
        # 通过第一个卷积层、Batch Normalization层和ReLU激活函数
        x = self.relu1(self.bn1(self.conv1(x)))
        # 通过第二个卷积层、Batch Normalization层和ReLU激活函数
        x = self.relu2(self.bn2(self.conv2(x)))

        # 通过第三个卷积层、Batch Normalization层和ReLU激活函数
        x = self.relu3(self.bn3(self.conv3(x)))

        # 通过第四个卷积层、Batch Normalization层和ReLU激活函数
        x = self.relu4(self.bn4(self.conv4(x)))

        # 通过第五个卷积层、Batch Normalization层和ReLU激活函数
        x = self.relu5(self.bn5(self.conv5(x)))

        # 通过策略头，计算策略（走法）输出
        policy = self.policy_head(x)

        # 通过价值头，计算价值输出
        value = self.value_head(x)

        return policy, value
