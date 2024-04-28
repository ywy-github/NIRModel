import os
import random
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorboard import summary
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from Main1.Metrics import all_metrics
from Main1.data_loader import MyData

# 定义自定义损失函数，加权二进制交叉熵
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight_positive):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, y_true, y_pred):
        y_true = y_true.to(dtype=torch.float32)
        loss = - (self.weight_positive * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return torch.mean(loss)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 10

    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 16
    epochs = 500
    learning_rate = 1e-5

    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/一期数据/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/一期数据/val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data

    test_benign_data = MyData("../data/一期数据/test/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/test/malignant", "malignant", transform=transform)
    test_data = test_benign_data + test_malignat_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=6,
                                 persistent_workers=True,
                                 pin_memory=True
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   persistent_workers=True,
                                   pin_memory=True
                                   )

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=6,
                             persistent_workers=True,
                             pin_memory=True
                             )



    model = models.alexnet(pretrained=True)
    #调整结构
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=9216, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=4096, out_features=1, bias=True),
        nn.Sigmoid()
  )

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    criterion = WeightedBinaryCrossEntropyLoss(2)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, amsgrad=False)

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        train_score = []
        train_pred = []
        train_targets = []
        total_train_loss = 0.0
        for batch in training_loader:
            images, targets, names = batch
            images = torch.cat([images] * 3, dim=1)
            images = images.to(device)
            # targets = targets.to(torch.float32)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(targets.view(-1, 1), output)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pred = (output >= 0.5).int().squeeze()

            train_score.append(output.cpu().detach().numpy())
            train_pred.extend(pred.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        model.eval()
        val_score = []
        val_pred = []
        val_targets = []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                images, targets, names = batch
                images = torch.cat([images] * 3, dim=1)
                # targets = targets.to(torch.float32)
                images = images.to(device)
                targets = targets.to(device)
                output = model(images)
                loss = criterion(targets.view(-1, 1), output)

                total_val_loss += loss.item()
                predicted_labels = (output >= 0.5).int().squeeze()

                val_score.append(output.flatten().cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        model.eval()
        test_score = []
        test_pred = []
        test_targets = []
        test_results = []
        total_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, targets, dcm_names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                output = model(data)

                loss = criterion(targets.view(-1, 1), output)
                total_test_loss += loss.item()
                predicted_labels = (output >= 0.5).int().squeeze()

                test_score.append(output.flatten().cpu().numpy())
                test_pred.extend(predicted_labels.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

                if ((epoch + 1) == 8):
                    for i in range(len(dcm_names)):
                        test_results.append({'dcm_name': dcm_names[i], 'pred': output[i].item(),
                                             'prob': predicted_labels[i].item(), 'label': targets[i].item()})

        if ((epoch + 1) == 8):
            # torch.save(model.state_dict(), "../models2/Vq-VAE-resnet18仅重构+分类器/Vq-VAE-resnet18仅重构+分类器-{}.pth".format(epoch + 1))
            # 记录每个样本的dcm_name、预测概率值和标签

            df = pd.DataFrame(test_results)
            filename = '../models2/excels/resnet18-8.xlsx'

            # 检查文件是否存在
            if not os.path.isfile(filename):
                # 如果文件不存在，创建新文件并保存数据到 Sheet1
                df.to_excel(filename, sheet_name='test', index=False)
            else:
                # 如果文件已经存在，打开现有文件并保存数据到 Sheet2
                with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                    df.to_excel(writer, sheet_name='test', index=False)

        print('%d epoch' % (epoch + 1))

        train_acc, train_sen, train_spe = all_metrics(train_targets, train_pred)

        train_score = np.concatenate(train_score)  # 将列表转换为NumPy数组
        train_targets = np.array(train_targets)
        train_auc = roc_auc_score(train_targets, train_score)

        print("训练集 acc: {:.4f}".format(train_acc) + " sen: {:.4f}".format(train_sen) +
              " spe: {:.4f}".format(train_spe) + " auc: {:.4f}".format(train_auc) +
              " loss: {:.4f}".format(total_train_loss))

        val_acc, val_sen, val_spe = all_metrics(val_targets, val_pred)

        val_score = np.concatenate(val_score)  # 将列表转换为NumPy数组
        val_targets = np.array(val_targets)
        val_auc = roc_auc_score(val_targets, val_score)

        print("验证集 acc: {:.4f}".format(val_acc) + " sen: {:.4f}".format(val_sen) +
              " spe: {:.4f}".format(val_spe) + " auc: {:.4f}".format(val_auc) +
              " loss: {:.4f}".format(total_val_loss))

        test_acc, test_sen, test_spe = all_metrics(test_targets, test_pred)

        test_score = np.concatenate(test_score)  # 将列表转换为NumPy数组
        test_targets = np.array(test_targets)
        test_auc = roc_auc_score(test_targets, test_score)

        print("测试集 acc: {:.4f}".format(test_acc) + " sen: {:.4f}".format(test_sen) +
              " spe: {:.4f}".format(test_spe) + " auc: {:.4f}".format(test_auc) +
              " loss: {:.4f}".format(total_test_loss))
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")