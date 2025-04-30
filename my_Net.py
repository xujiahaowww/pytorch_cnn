import torch
import torch.nn as nn
import copy
from typing import Any


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 128, 3),
            nn.AdaptiveAvgPool2d((100, 100)),
            nn.LeakyReLU(inplace=True),
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.AdaptiveAvgPool2d((50, 50)),
            nn.LeakyReLU(inplace=True),
        )
        self.fn = nn.Sequential(
            nn.Linear(64 * 50 * 50, 256),
            nn.LeakyReLU(inplace=True)
        )
        self.outFn = nn.Linear(256, 3)

        self.reChange = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.AdaptiveAvgPool2d((50, 50)),
        )

    def forward(self, value) -> Any:
        # 3通道图片
        colony = copy.deepcopy(value)
        value = self.features_1(value)
        value = self.features_2(value)
        # 添加输入值，防止梯度消失，原特征图已成64通道 50*50
        colony = self.reChange(colony)

        value += colony

        value = value.view(value.size(0), -1)
        value = self.fn(value)
        out = self.outFn(value)
        return out
