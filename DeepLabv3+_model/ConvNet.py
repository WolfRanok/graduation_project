import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation_rate, num_layers):
        """
        :param input_size: 输入特征数量
        :param hidden_size: 隐藏层特征数量
        :param kernel_size: 卷积核大小
        :param dilation_rate: 空洞卷积的扩张率
        :param num_layers: 网络层数
        """
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size,kernel_size=kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    # 示例参数


input_size = 1  # 输入特征数量
hidden_size = 64  # 隐藏层特征数量
kernel_size = 3  # 卷积核大小
dilation_rate = 2  # 空洞卷积的扩张率
num_layers = 3  # 网络层数

# 创建模型实例
model = TemporalConvNet(input_size, hidden_size, kernel_size, dilation_rate, num_layers)

# 数据输入 (batch_size, seq_length, input_size)
batch_size = 32
seq_length = 100
x = torch.randn(batch_size, seq_length, input_size)

# 前向传播
output = model(x)
print(output.shape)  # 输出形状应该是 (batch_size, seq_length, hidden_size)