import time

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from six.moves import xrange

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import torch.utils.data as data
from torchvision import models
import numpy as np
from PIL import Image
import glob
import random

from Main1.Metrics import all_metrics
from Main1.data_loader import MyData

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.path(x)
        return x

class Model(nn.Module):
    def __init__(self,encoder):
        super(Model, self).__init__()

        self._encoder = encoder
        # self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
        #                               out_channels=embedding_dim,
        #                               kernel_size=1,
        #                               stride=1)


        self.classifier = Classifier(512*14*14,512,1)

    def forward(self, x):
        z = self._encoder(x)

        classifier_outputs = self.classifier(z.view(z.size(0),-1))


        return classifier_outputs

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

    seed = 64

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
    epochs = 1000

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
                                 num_workers=1,
                                 persistent_workers=True,
                                 pin_memory=True
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   persistent_workers=True,
                                   pin_memory=True
                                  )

    test_loader = DataLoader(test_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   persistent_workers=True,
                                   pin_memory=True
                                   )


    #设置encoder
    resnet18 = models.resnet18(pretrained=True)

    for param in resnet18.parameters():
        param.requires_grad = False

    for name, param in resnet18.named_parameters():
        if "layer3" in name:
            param.requires_grad = True
        if "layer4" in name:
            param.requires_grad = True
        if "fc" in name:
            param.requires_grad = True

    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    model = Model(resnet18).to(device)

    criterion = WeightedBinaryCrossEntropyLoss(2)

    criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, amsgrad=False)

    # scheduler = StepLR(optimizer,10,0.1)
    train_res_recon_error = []
    train_res_perplexity = []

    val_res_recon_error = []
    val_res_perplexity = []

    test_res_recon_error = []
    test_res_perplexity = []


    for epoch in range(epochs):
        # if((epoch+1) % 10 == 0):
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.8
        model.train()
        train_score = []
        train_pred = []
        train_targets = []
        total_train_loss = 0.0
        for batch in training_loader:
            data, targets, dcm_names = batch
            data = torch.cat([data] * 3, dim=1)
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            classifier_outputs = model(data)
            classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)

            classifier_loss.backward()
            optimizer.step()
            # scheduler.step()

            predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
            train_score.append(classifier_outputs.cpu().detach().numpy())
            train_pred.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            total_train_loss += classifier_loss
        # writer.add_scalar('Loss/Train', total_train_loss, epoch)
        val_score = []
        val_pred = []
        val_targets = []
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                classifier_outputs = model(data)

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                val_score.append(classifier_outputs.flatten().cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

                total_val_loss += classifier_loss

        test_score = []
        test_pred = []
        test_targets = []
        total_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                classifier_outputs = model(data)


                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                test_score.append(classifier_outputs.flatten().cpu().numpy())
                test_pred.extend(predicted_labels.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

                total_test_loss += classifier_loss

        # writer.add_scalar('Loss/Val', total_val_loss, epoch)

        if ((epoch + 1) == 18):
            torch.save(model.state_dict(),"../models1/package/resnet18-{}.pth".format(epoch + 1))
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

