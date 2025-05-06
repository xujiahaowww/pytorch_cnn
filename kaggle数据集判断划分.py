from typing import Any
import torch
import torch.nn as nn
from my_Net import Net
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image


# 获取训练集和测试集
def getData() -> tuple[Any, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize((100, 100))
    ])
    # 训练数据集和加载器
    dataSet_train = datasets.ImageFolder('../day03/flowers/flowers', transform=transform)
    dataSet_train_DataLoader = DataLoader(dataSet_train, batch_size=50, shuffle=True)
    # 加载excel文件使用到classes
    classes = dataSet_train.classes
    print(classes)

    # 测试数据集和加载器
    dataSet_test = datasets.ImageFolder('../day03/flowers/flowers', transform=transform)
    dataSet_test_DataLoader = DataLoader(dataSet_test, batch_size=50, shuffle=True)
    return classes, dataSet_train_DataLoader, dataSet_test_DataLoader


# 创建训练方法
def train(model, classes, dataSet_train_DataLoader) -> None:
    model.train()
    # 判断cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    tbpath = os.path.realpath(os.path.join(os.path.dirname(__file__), 'tensorboard'))
    writer = SummaryWriter(log_dir=tbpath)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(25):
        loss_all = 0
        success_all = 0

        # 创建np模板
        data_arr = np.empty(shape=(0, 7))
        np.set_printoptions(suppress=True)  # 取消科学计数法
        class_name = classes
        for i, (train_x, train_y) in enumerate(dataSet_train_DataLoader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            pred = model(train_x)
            loss = lossFn(pred, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_all += loss.item()
            _, predicted = torch.max(pred, 1)
            success_all += (predicted == train_y).sum().item()

            # if i % 10 == 0:
            #     print('第几轮的批次', i + 1)
            #     grid = make_grid(train_x)
            #     writer.add_image(f'批次图片展示第{i + 1}批次', grid, 10)

            y1 = pred.detach().cpu().numpy()
            y2 = np.expand_dims(predicted.detach().cpu().numpy(), 1)
            y3 = np.expand_dims(train_y.detach().cpu().numpy(), 1)
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


def test(model, dataSet_test_DataLoader) -> None:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lossFn = nn.CrossEntropyLoss()
    loss_all = 0
    success_all = 0
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(dataSet_test_DataLoader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pred = model(x_test)
            loss = lossFn(pred, y_test)
            loss_all += loss.item()
            success_all += (pred == y_test).sum().item()

        print(f'测试损失率：{loss_all / len(dataSet_test_DataLoader)}')
        print(f'测试成功率：{success_all / len(dataSet_test_DataLoader.dataset)}')


def save(model) -> None:
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'mode_save', 'kaggle.pth'))
    torch.save(model.state_dict(), path)


def save_onnx(model):
    mymodel = model()
    save_dict = torch.load('../model_save/kaggle.pth')
    mymodel.load_state_dict(save_dict)
    print(mymodel)
    onnxpath = os.path.join(os.path.dirname(__file__), "onnx", 'kaggle.onnx')
    print(onnxpath)
    # 创建一个实例输入
    x = torch.randn(1, 3, 100, 100)
    # 导出onnx
    torch.onnx.export(
        mymodel,
        x,
        onnxpath,
        verbose=False,  # 输出转换过程
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    print("onnx导出成功")


def predict(img_path, model) -> None:
    model.eval()

    save_dict = torch.load('./kaggle.pth')
    model.load_state_dict(save_dict)

    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100))
    ])
    t_img = transform(img).unsqueeze(0)
    print(t_img.shape)

    with torch.no_grad():
        output = model(t_img)
        print(output.data)
        _, predicted = torch.max(output.data, 1)

        print(predicted.item())


if __name__ == '__main__':
    model = Net((3, 6, 3, 3), (6, 12, 1, None), (12, 24, (3, 1), None), (24, 30, (1, 3), None))
    # classes, dataSet_train_DataLoader, dataSet_test_DataLoader = getData()
    # train(model, classes, dataSet_train_DataLoader)
    # test(model, dataSet_test_DataLoader)
    # save(model)
    # save_onnx(model)
    predict('./test.jpg', model)
