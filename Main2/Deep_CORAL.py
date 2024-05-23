from __future__ import print_function

import time
from itertools import cycle

import matplotlib.pyplot as plt
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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.b2(x)
        x = self.b3(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )



    def forward(self, inputs):
        x = self.deconv1(inputs)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x

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
    def __init__(self,encoder,num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = encoder
        # self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
        #                               out_channels=embedding_dim,
        #                               kernel_size=1,
        #                               stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder()

    def forward(self, x):
        z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

class ExtendedModel(nn.Module):
    def __init__(self, model):
        super(ExtendedModel, self).__init__()
        self.model = model
        self.classifier = Classifier(512*7*7,512,1)

    def forward(self, x):
        z = self.model._encoder(x)
        loss, quantized, perplexity, _ = self.model._vq_vae(z)
        classifier_output = self.classifier(quantized.view(quantized.size(0),-1))
        x_recon = self.model._decoder(quantized)


        return loss, quantized,x_recon, perplexity, classifier_output


# 定义联合模型的损失函数
def joint_loss_function(recon_loss, vq_loss, classifier_loss,adaptation_loss ,lambda_recon, lambda_vq,
                                             lambda_classifier,lambda_adaptation):
    # 总损失
    total_loss = lambda_recon * recon_loss + lambda_vq*vq_loss \
                 + lambda_classifier * classifier_loss+adaptation_loss*lambda_adaptation

    return total_loss

# 定义自定义损失函数，加权二进制交叉熵
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight_positive):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, y_true, y_pred):
        y_true = y_true.to(dtype=torch.float32)
        loss = - (self.weight_positive * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return torch.mean(loss)


class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source, target):
        d = source.size(1)  # 特征的维度
        source_cov = self.compute_covariance(source)
        target_cov = self.compute_covariance(target)
        # 计算源域和目标域协方差矩阵之间的差异
        loss = torch.mean((source_cov - target_cov) ** 2) / (4 * d * d)
        return loss

    def compute_covariance(self, features):
        n = features.size(0)  # 样本数量
        mean = torch.mean(features, dim=0, keepdim=True)
        features = features - mean

        features = features - mean
        cov = features.t().mm(features) / (n - 1)
        return cov


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

    embedding_dim = 64
    num_embeddings = 512  # 和encoder输出维度相同，和decoder输入维度相同

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-5

    lambda_recon = 0.2
    lambda_vq = 0.2
    lambda_classifier = 0.4
    lambda_adaptation = 0.2

    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    source_train_benign_data = MyData("../data/一期all/benign", "benign", transform=transform)
    source_train_malignat_data = MyData("../data/一期all/malignant", "malignant", transform=transform)
    source_train_data = source_train_benign_data + source_train_malignat_data

    target_train_benign_data = MyData("../data/qc后二期数据/train/wave1/benign", "benign", transform=transform)
    target_train_malignat_data = MyData("../data/qc后二期数据/train/wave1/malignant", "malignant", transform=transform)
    target_train_data = target_train_benign_data + target_train_malignat_data


    val_benign_data = MyData("../data/qc后二期数据/val/wave1/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/qc后二期数据/val/wave1/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data

    test_benign_data = MyData("../data/qc后二期数据/test/wave1/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/qc后二期数据/test/wave1/malignant", "malignant", transform=transform)
    test_data = test_benign_data + test_malignat_data

    source_training_loader = DataLoader(source_train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 persistent_workers=True,
                                 pin_memory=True
                                 )

    target_training_loader = DataLoader(target_train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 persistent_workers=True,
                                 pin_memory=True
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   persistent_workers=True,
                                   pin_memory=True
                                  )

    test_loader = DataLoader(test_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   persistent_workers=True,
                                   pin_memory=True
                                   )

    model = torch.load("../models2/筛查重构/VQ-VAE-筛查重构-200.pth", map_location=device)

    for param in model.parameters():
        param.requires_grad = True
    # for name, param in model.named_parameters():
    #     if "6" in name:
    #         param.requires_grad = True
    #     if "7" in name:
    #         param.requires_grad = True
    #     if "_vq_vae" in name:
    #         param.requires_grad = True
    #     if "_decoder" in name:
    #         param.requires_grad = True

    extendModel = ExtendedModel(model).to(device)

    coral_loss = CoralLoss()
    criterion = WeightedBinaryCrossEntropyLoss(1.6)  # 调整这个权重以提高对灵敏度的重视
    criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, extendModel.parameters()), lr=learning_rate, amsgrad=False)

    for epoch in range(epochs):
        extendModel.train()
        train_score = []
        train_pred = []
        train_targets = []
        total_train_loss = 0.0
        train_classifier_loss = 0.0
        train_res_recon_error = 0.0
        train_res_perplexity = 0.0

        source_iter = cycle(source_training_loader)
        target_iter = cycle(target_training_loader)
        for i in range(len(source_training_loader)):
            source_data, source_labels,_ = next(source_iter)
            target_data, _,_ = next(target_iter)

            source_data = torch.cat([source_data] * 3, dim=1)
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)

            target_data = torch.cat([target_data] * 3, dim=1)
            target_data = target_data.to(device)

            optimizer.zero_grad()

            vq_loss, source_features, source_data_recon, perplexity, classifier_outputs = extendModel(source_data)
            _, target_features, _, _, _ = extendModel(target_data)

            classifier_loss = criterion(source_labels.view(-1, 1), classifier_outputs)
            adaptation_loss = coral_loss(source_features.view(source_features.size(0), -1),
                                         target_features.view(target_features.size(0), -1))

            data_variance = torch.var(source_data)
            recon_loss = F.mse_loss(source_data_recon, source_data) / data_variance

            total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss,adaptation_loss,lambda_recon, lambda_vq,
                                             lambda_classifier,lambda_adaptation)
            total_loss.backward()
            optimizer.step()
            # scheduler.step()

            predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
            train_score.append(classifier_outputs.cpu().detach().numpy())
            train_pred.extend(predicted_labels.cpu().numpy())
            train_targets.extend(source_labels.cpu().numpy())

            total_train_loss += total_loss
            train_classifier_loss += classifier_loss
            train_res_recon_error += recon_loss
            train_res_perplexity += perplexity
        # writer.add_scalar('Loss/Train', total_train_loss, epoch)
        val_score = []
        val_pred = []
        val_targets = []
        extendModel.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                _,_,_,_, classifier_outputs = extendModel(data)

                predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
                val_score.append(classifier_outputs.flatten().cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        test_score = []
        test_pred = []
        test_targets = []
        extendModel.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                _,_,_,_, classifier_outputs = extendModel(data)

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                test_score.append(classifier_outputs.flatten().cpu().numpy())
                test_pred.extend(predicted_labels.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        # 训练的时候注释掉下边保存模型的代码，之后二次训练根据选择的epoch来保存模型

        # if ((epoch + 1) == 64 or (epoch + 1) == 66):
        #     torch.save(extendModel.state_dict(), "../models3/筛查重构+分类联合学习/筛查重构+分类联合学习-{}.pth".format(epoch + 1))

        print('%d epoch' % (epoch + 1))
        train_acc, train_sen, train_spe = all_metrics(train_targets, train_pred)

        train_score = np.concatenate(train_score)  # 将列表转换为NumPy数组
        train_targets = np.array(train_targets)
        train_auc = roc_auc_score(train_targets, train_score)

        print("训练集 acc: {:.4f}".format(train_acc) + " sen: {:.4f}".format(train_sen) +
              " spe: {:.4f}".format(train_spe) + " auc: {:.4f}".format(train_auc) +
              " loss: {:.4f}".format(train_classifier_loss) + " train_recon_loss: {:.4f}".format(train_res_recon_error)
              + " train_perplexity: {:.4f}".format(train_res_perplexity))

        val_acc, val_sen, val_spe = all_metrics(val_targets, val_pred)

        val_score = np.concatenate(val_score)  # 将列表转换为NumPy数组
        val_targets = np.array(val_targets)
        val_auc = roc_auc_score(val_targets, val_score)

        print("验证集 acc: {:.4f}".format(val_acc) + " sen: {:.4f}".format(val_sen) +
              " spe: {:.4f}".format(val_spe) + " auc: {:.4f}".format(val_auc))

        test_acc, test_sen, test_spe = all_metrics(test_targets, test_pred)

        test_score = np.concatenate(test_score)  # 将列表转换为NumPy数组
        test_targets = np.array(test_targets)
        test_auc = roc_auc_score(test_targets, test_score)

        print("测试集 acc: {:.4f}".format(test_acc) + " sen: {:.4f}".format(test_sen) +
              " spe: {:.4f}".format(test_spe) + " auc: {:.4f}".format(test_auc))
