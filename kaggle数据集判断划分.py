from typing import Any
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report


# 获取训练集和测试集
def getData() -> tuple[Any, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize((200, 200))
    ])
    # 训练数据集和加载器
    dataSet_train = datasets.ImageFolder('../day04/kaggle_flower', transform=transform)
    dataSet_train_DataLoader = DataLoader(dataSet_train, batch_size=32, shuffle=True)
    # 加载excel文件使用到classes
    classes = dataSet_train.classes

    # 测试数据集和加载器
    dataSet_test = datasets.ImageFolder('../day04/kaggle_flower', transform=transform)
    dataSet_test_DataLoader = DataLoader(dataSet_train, batch_size=32, shuffle=True)
    return classes, dataSet_train_DataLoader, dataSet_test_DataLoader



if __name__ == '__main__':
    getData()
