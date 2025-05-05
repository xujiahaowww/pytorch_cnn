from typing import Any
import torch
import torch.nn as nn
from my_Net import Net
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
    dataSet_test_DataLoader = DataLoader(dataSet_test, batch_size=32, shuffle=True)
    return classes, dataSet_train_DataLoader, dataSet_test_DataLoader


# 创建训练方法
def train(model, classes, dataSet_train_DataLoader) -> None:
    model.train()
    tbpath = os.path.realpath(os.path.join(os.path.dirname(__file__), 'tensorboard'))
    writer = SummaryWriter(log_dir=tbpath)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        loss_all = 0
        success_all = 0

        # 创建np模板
        data_arr = np.empty(shape=(0, 5))
        np.set_printoptions(suppress=True)  # 取消科学计数法
        class_name = classes
        for i, (train_x, train_y) in enumerate(dataSet_train_DataLoader):
            pred = model(train_x)
            loss = lossFn(pred, train_y)
            optimizer.step()
            optimizer.zero_grad()

            loss_all += loss.item()

            _, predicted = torch.max(pred, 1)
            success_all += (predicted == train_y).sum().item()

            y1 = pred.detach().numpy()
            y2 = np.expand_dims(predicted.detach().numpy(), 1)
            y3 = np.expand_dims(train_y.detach().numpy(), 1)

            y_finall = np.hstack((y1, y2, y3), dtype=float)
            data_arr = np.vstack((data_arr, y_finall))
        print(f'第{epoch + 1}次训练的损失率：{loss_all / len(dataSet_train_DataLoader)}')
        print(f'第{epoch + 1}次训练的成功率：{success_all / len(dataSet_train_DataLoader.dataset)}')

        writer.add_scalar("Loss/train", loss_all / len(dataSet_train_DataLoader), epoch)
        writer.add_scalar("Accuracy/train", success_all / len(dataSet_train_DataLoader.dataset), epoch)

        # 存储 csv文件
        label = [*class_name, 'predict', 'real']
        data_excel = pd.DataFrame(data_arr, columns=label)
        path_name = os.path.join(os.getcwd(), 'excel', f'第{epoch + 1}次数据集.csv')
        data_excel.to_csv(path_name, index=False)

        read_data = pd.read_csv(path_name)
        true_label = read_data["real"].values
        pre_label = read_data["predict"].values
        # 查看精确率和召回率
        report = classification_report(y_true=true_label, y_pred=pre_label)
        print(report)

    writer.close()  # 关闭写入


def test(model,dataSet_test_DataLoader):
    model.eval()
    lossFn = nn.CrossEntropyLoss()
    loss_all = 0
    success_all = 0
    for i, (x_test,y_test) in enumerate(dataSet_test_DataLoader):
        pred = model(x_test)
        loss = lossFn(pred, y_test)
        loss_all += loss.item()
        success_all += (pred == y_test).sum().item()

    print(f'测试损失率：{loss_all / len(dataSet_test_DataLoader)}')
    print(f'测试成功率：{success_all / len(dataSet_test_DataLoader.dataset)}')

if __name__ == '__main__':
    model = Net()
    classes, dataSet_train_DataLoader, dataSet_test_DataLoader = getData()
    train(model, classes, dataSet_train_DataLoader)
    test(model, dataSet_test_DataLoader)
