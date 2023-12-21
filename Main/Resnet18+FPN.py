import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorboard import summary
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from Metrics import all_metrics
from data_loader import MyData
import SKnet as SK

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CSFblock(nn.Module):
    ###联合网络
    def __init__(self, in_channels, channels_1, strides):
        super().__init__()
        # self.layer = nn.Conv1d(in_channels, 512, kernel_size = 1, padding = 0, dilation = 1)
        self.Up = nn.Sequential(
            # nn.MaxPool2d(kernel_size = int(strides*2+1), stride = strides, padding = strides),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=strides, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.Fgp = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            oneConv(in_channels, channels_1, 1, 0, 1),
            oneConv(channels_1, in_channels, 1, 0, 1),

        )
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_h, x_l):
        x1 = x_h
        x2 = self.Up(x_l)

        x_f = x1 + x2
        # print(x_f.size())
        Fgp = self.Fgp(x_f)
        # print(Fgp.size())
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1, x_se2], 2)
        x_se = self.softmax(x_se)
        att_3 = torch.unsqueeze(x_se[:, :, 0], 2)
        att_5 = torch.unsqueeze(x_se[:, :, 1], 2)
        x1 = att_3 * x1
        x2 = att_5 * x2
        x_all = x1 + x2
        return x_all


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.CSF1 = CSFblock(128,32,2)
        self.CSF2 = CSFblock(128,32,2)
        self.GAP =  nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x3 = x3
        x2 = self.CSF1(x2,x3)
        x1 = self.CSF2(x1,x2)
        x3 = self.GAP(x3).squeeze(-1).squeeze(-1)
        x2 = self.GAP(x2).squeeze(-1).squeeze(-1)
        x1 = self.GAP(x1).squeeze(-1).squeeze(-1)
        results = torch.cat([x3,x2,x1],1)
        return results

class CF(nn.Module):
    def __init__(self, class_num):
        super(CF, self).__init__()
        self.backbone = SK.SKNet26(2)
        self.FPN = FPN()
        self.fc = nn.Sequential(
                  nn.Linear(384, 256),
                  nn.Dropout(0.1),
                  nn.ReLU(),
                  nn.Linear(256, 2),
                  )
    def forward(self, x):
        feature = self.backbone(x)
        #print(feature.size())
        feature = self.FPN(feature)
        results = self.fc(feature)
        return feature,results

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

    batch_size = 8
    epochs = 1000
    learning_rate = 1e-4

    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train+clahe/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train+clahe/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/一期数据/new_val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/一期数据/new_val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data


    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=8,
                                 persistent_workers=True)

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=8,
                                   persistent_workers=True
                                  )



    model = models.resnet18(pretrained=True)
    num_hidden = 256
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_hidden, num_hidden//2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_hidden//2, 1),
        nn.Sigmoid()
    )

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "layer3" in name:
            param.requires_grad = True
        if "layer4" in name:
            param.requires_grad = True
        if "fc" in name:
            param.requires_grad = True

    criterion = WeightedBinaryCrossEntropyLoss(2)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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

        # if ((epoch + 1)%50 == 0):
        #     torch.save(model, "../models/result/resnet18-resize448{}.pth".format(epoch + 1))

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


    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")