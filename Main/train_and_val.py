import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
from Main.Metrics import all_metrics
from Main.data_loader import MyData
from tqdm import tqdm

#创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16384, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
    # 读取数据
    train_benign_data = MyData("../data/train/benign", "benign",transform=transform)
    train_malignat_data = MyData("../data/train/malignant", "malignant",transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/val/benign", "benign",transform=transform)
    val_malignat_data = MyData("../data/val/malignant", "malignant",transform=transform)
    val_data = val_benign_data + val_malignat_data

    # 利用 DataLoader 来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=64)
    val_dataloader = DataLoader(val_data, batch_size=64)

    tudui = Tudui()
    if torch.cuda.is_available():
        tudui = tudui.cuda()

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    # 优化器
    # learning_rate = 0.01
    # 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录验证的次数
    total_val_step = 0
    # 训练的轮数
    epoch = 100

    for i in range(epoch):

        # 训练步骤开始
        tudui.train()
        train_predictions = []
        train_targets = []
        total_train_loss = 0.0
        for data in train_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            total_train_loss += loss.item()

        train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
        if((i+1)%10==0):
            print("训练集上的准确率: {:.4f}".format(train_acc))
            print("训练集上的灵敏度: {:.4f}".format(train_sen))
            print("训练集上的特异度: {:.4f}".format(train_spe))
            print("总的训练集上的损失:{}".format(total_train_loss))

        val_predictions = []
        val_targets = []
        total_val_loss = 0.0
        # 验证步骤开始
        tudui.eval()
        with torch.no_grad():
            for data in val_dataloader:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = tudui(imgs)
                loss = loss_fn(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                total_val_loss += loss.item()
            if ((i + 1) % 10 == 0):
                val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
                print("验证集上的准确率: {:.4f}".format(val_acc))
                print("验证集上的灵敏度: {:.4f}".format(val_sen))
                print("验证集上的特异度: {:.4f}".format(val_spe))
                print("总的验证集上的损失:{}".format(total_val_loss))


