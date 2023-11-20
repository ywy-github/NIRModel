from __future__ import print_function

import time

import matplotlib.pyplot as plt
from six.moves import xrange

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import glob
import random

from Metrics import all_metrics
from data_loader import MyData


# 定义自定义损失函数，加权二进制交叉熵
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight_positive):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, y_true, y_pred):
        y_true = y_true.to(dtype=torch.float32)
        loss = - (self.weight_positive * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return torch.mean(loss)

# 定义 Focal Loss
class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.55, gamma=2.0):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        labels = labels.to(dtype=torch.float32)
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    epochs = 500

    commitment_cost = 0.25

    decay = 0.99

    weight_positive = 2  # 调整这个权重以提高对灵敏度的重视

    learning_rate = 1e-5


    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    test_benign_data = MyData("../data/一期数据/test/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/test/malignant", "malignant", transform=transform)
    test_data = test_benign_data + test_malignat_data

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

    model = torch.load("../models/VQ-Resnet/resnet18-resize512.pth", map_location=device)

    criterion = WeightedBinaryCrossEntropyLoss(2)
    criterion.to(device)

    test_predictions = []
    test_targets = []
    test_results = []
    total_test_loss = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, targets, dcm_names = batch
            data = torch.cat([data] * 3, dim=1)
            data = data.to(device)
            targets = targets.to(device)
            classifier_outputs = model(data)

            loss = criterion(targets.view(-1, 1), classifier_outputs)

            predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
            # 记录每个样本的dcm_name、预测概率值和标签
            for i in range(len(dcm_names)):
                test_results.append({'dcm_name': dcm_names[i], 'pred': classifier_outputs[i].item(),
                                     'prob': predicted_labels[i].item(), 'label': targets[i].item()})
            test_predictions.extend(predicted_labels.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            total_test_loss.append(loss.item())

            # concat = torch.cat((data[0].view(128, 128),
            #                     data_recon[0].view(128, 128)), 1)
            # plt.matshow(concat.cpu().detach().numpy())
            # plt.show()

    train_acc, train_sen, train_spe = all_metrics(test_targets, test_predictions)

    print("测试集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
          "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(np.mean(total_test_loss[-10:])))

    df = pd.DataFrame(test_results)
    # df.to_excel("../models/result/resnet18-data2-train.xlsx", index=False)
