from __future__ import print_function

import time

import matplotlib.pyplot as plt
from six.moves import xrange

import torch.nn as nn
import torch.nn.functional as F
from tensorboard import summary
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

from Main.Metrics import all_metrics
from Main.data_loader import MyData

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.b1=nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                         nn.BatchNorm2d(64), nn.ReLU(),
                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)),
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2)),
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2)),
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2)),
        self.net = nn.Sequential(*self.b1, *self.b2, *self.b3, *self.b4, *self.b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, 1),nn.Sigmoid())

    def forward(self, x):
        output = self.net(x)
        return output

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.60, gamma=2.0):
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

    batch_size = 64
    epochs = 100
    learning_rate = 1e-5

    # 读取数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/一期数据/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/一期数据/val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    validation_loader = DataLoader(val_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)



    model = Model().to(device)

    criterion = Focal_Loss()
    criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_time = time.time()  # 记录训练开始时间
    writer = SummaryWriter("../Logs")
    for epoch in range(epochs):
        model.train()
        train_predictions = []
        train_targets = []
        total_train_loss = 0
        for batch in training_loader:
            images, targets,names= batch
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,targets.view(-1,1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted_labels = (output >= 0.5).int().squeeze()
            train_predictions.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        writer.add_scalar('Loss/Train', total_train_loss, epoch)

        model.eval()
        val_predictions = []
        val_targets = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                images, targets, names = batch
                images = images.to(device)
                targets = targets.to(device)
                output = model(images)
                loss = criterion(output, targets.view(-1,1))
                total_val_loss += loss.item()
                predicted_labels = (output >= 0.5).int().squeeze()
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        writer.add_scalar('Loss/Val', total_val_loss, epoch)
        if ((epoch + 1) % 1 == 0):
            # torch.save(models, "VQ_VAE{}.pth".format(i+1))
            print('%d epoch' % (epoch + 1))

            train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
            print("训练集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
                  "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(total_train_loss))

            val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
            print("验证集 acc: {:.4f}".format(val_acc) + "sen: {:.4f}".format(val_sen) +
                  "spe: {:.4f}".format(val_spe) + "loss: {:.4f}".format(total_val_loss))

    # 结束训练时间
    writer.close()
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")
