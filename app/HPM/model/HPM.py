import torch
import torch.nn as nn
from torchsummary import summary

from model.models import *


class HPM(nn.Module):
    def __init__(self):
        super(HPM, self).__init__()
        self.layer1 = MLP()

    def forward(self, x):
        return self.layer1(x)


if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = HPM().to(device)
    # 打印模型的参数量大小、每个层的结构和参数量，以及模型需要的GFLOPS
    summary(model, input_size=(4,), device=device.type)