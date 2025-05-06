import torch
import torch.nn as nn
import copy
from typing import Any


class Net(nn.Module):
    def __init__(self, *args):
        super(Net, self).__init__()
        self.Conv2dArr = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=i[0],
                    out_channels=i[1],
                    kernel_size=i[2],
                    stride=1,
                    padding=0,
                    groups=3 if i[3] is not None else 1
                )
                for i in args
            ]
        )
        print(self.Conv2dArr)
        self.features_1 = nn.Sequential(
            self.Conv2dArr[0],  # 3 6 3 3
            self.Conv2dArr[1],  # 6, 12, 1, None
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(50),
        )
        self.features_2 = nn.Sequential(
            self.Conv2dArr[2],  # 12, 24, (3, 1), None
            self.Conv2dArr[3],  # 24, 30, (1, 3), None
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(30),
        )

        self.fn = nn.Sequential(
            nn.BatchNorm1d(30 * 30 * 30),
            nn.ReLU(),
            nn.Linear(30 * 30 * 30, 5),

        )

        # self.outFn = nn.Linear(256, 3)

        self.reChange = nn.Sequential(
            nn.Conv2d(3, 30, 3),
            nn.AdaptiveAvgPool2d(30)
        )

    def forward(self, value) -> Any:
        # 3通道图片
        # colony = copy.deepcopy(value)
        value = self.features_1(value)
        value = self.features_2(value)
        # 添加输入值，防止梯度消失，原特征图已成40通道 50*50
        # colony = self.reChange(colony)
        # value += colony

        value = value.view(value.size(0), -1)
        value = self.fn(value)
        # out = self.outFn(value)
        return value
