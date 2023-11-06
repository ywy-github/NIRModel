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
from Main.Metrics import all_metrics
from Main.data_loader import MyData

#通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#先接通道注意力模块、再接空间注意力模块
class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x*self.ca(x)
        result = out*self.sa(out)
        return result


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

    batch_size = 64
    epochs = 1000
    learning_rate = 1e-4

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



    model = models.resnet18(pretrained=True)

    #调整结构
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    CBAM = CBAM(model.fc.in_features)
    model.layer4.add_module('CBAM', CBAM)
    num_hidden = 256
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden, 1),
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
    writer = SummaryWriter("../Logs")
    for epoch in range(epochs):
        model.train()
        train_score = []
        train_pred = []
        train_targets = []
        total_train_loss = 0.0
        for batch in training_loader:
            images, targets, names= batch
            images = images.to(device)
            # targets = targets.to(torch.float32)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(targets.view(-1, 1),output)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pred = (output >= 0.5).int().squeeze()

            train_score.append(output.cpu().detach().numpy())
            train_pred.extend(pred.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        writer.add_scalar('Loss/Train', total_train_loss, epoch)

        model.eval()
        val_score = []
        val_pred = []
        val_targets = []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                images, targets, names = batch
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
        writer.add_scalar('Loss/Val', total_val_loss, epoch)

        if ((epoch + 1) % 50 == 0):
            torch.save(model, "../models/resnet/resnet18{}.pth".format(epoch+1))
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
                  " spe: {:.4f}".format(val_spe) +" auc: {:.4f}".format(val_auc)+
                  " loss: {:.4f}".format(total_val_loss))


    writer.close()
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")