from __future__ import print_function

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
from Main1.data_loader import DoubleTreeChannels


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
    def __init__(self,encoder1,encoder2,num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder1 = encoder1
        self._encoder2 = encoder2
        # self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
        #                               out_channels=embedding_dim,
        #                               kernel_size=1,
        #                               stride=1)
        if decay > 0.0:
            self._vq_vae1 = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
            self._vq_vae2 = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae1 = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
            self._vq_vae2 = VectorQuantizer(num_embeddings, embedding_dim,
                                            commitment_cost)

        self.classifier = Classifier(200704,512,1)

        self._decoder1 = Decoder()
        self._decoder2 = Decoder()

        self.Avg = nn.AdaptiveMaxPool2d(1)

    def forward(self, data1,data2):
        z1 = self._encoder1(data1)
        z2 = self._encoder2(data2)

        # z = self._pre_vq_conv(z)
        loss1, quantized1, perplexity1, _ = self._vq_vae1(z1)
        loss2, quantized2, perplexity2, _ = self._vq_vae2(z2)
        quantized = torch.cat([quantized1, quantized2], dim=1)

        feature = quantized.view(quantized.size(0), -1)

        # 拼接到展平后的特征上
        # combined_features = torch.cat((feature,one_hot_cup_sizes), dim=1)

        classifier_outputs = self.classifier(feature)

        x_recon1 = self._decoder1(quantized1)
        x_recon2 = self._decoder2(quantized2)

        return loss1,loss2,x_recon1,x_recon2,perplexity1,perplexity2,classifier_outputs

# 定义联合模型的损失函数
def joint_loss_function(recon_loss1,recon_loss2, vq_loss1,vq_loss2, classifier_loss,
                        lambda_recon1,lambda_recon2, lambda_vq1,lambda_vq2,lambda_classifier):
    # 总损失
    total_loss = lambda_recon1 * recon_loss1 + lambda_vq1 * vq_loss1 + lambda_classifier * classifier_loss + \
                 lambda_recon2 * recon_loss2 + lambda_vq2 * vq_loss2

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

# 定义 Focal Loss
class WeightedBinaryCrossEntropyLossWithRegularization(nn.Module):
    def __init__(self, weight_positive, lambda_reg):
        super(WeightedBinaryCrossEntropyLossWithRegularization, self).__init__()
        self.weight_positive = weight_positive
        self.lambda_reg = lambda_reg  # 正则化系数

    def forward(self, y_true, y_pred, model):
        y_true = y_true.to(dtype=torch.float32)
        bce_loss = - (self.weight_positive * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        bce_loss = torch.mean(bce_loss)

        # 添加L2正则化项
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=2)

        total_loss = bce_loss + self.lambda_reg * reg_loss

        return total_loss
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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

    embedding_dim = 64
    num_embeddings = 512  # 和encoder输出维度相同，和decoder输入维度相同

    commitment_cost = 0.25

    decay = 0.99


    learning_rate = 1e-5

    lambda_recon1 = 0.1
    lambda_vq1 = 0.1
    lambda_classifier = 0.6

    lambda_recon2 = 0.1
    lambda_vq2 = 0.1


    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    fold_data = "二期数据CLAHE"

    train_benign_data = DoubleTreeChannels("../data/"+fold_data+"/train/wave1/benign",
                                                           "../data/"+fold_data+"/train/wave2/benign",
                                                           "../data/"+fold_data+"/train/wave3/benign",
                                                           "../data/"+fold_data+"/train/wave4/benign",
                                                           "benign",
                                                           transform=transform)

    train_malignant_data = DoubleTreeChannels(
        "../data/"+fold_data+"/train/wave1/malignant",
        "../data/"+fold_data+"/train/wave2/malignant",
        "../data/"+fold_data+"/train/wave3/malignant",
        "../data/"+fold_data+"/train/wave4/malignant",
        "malignant",
        transform=transform)

    train_data = train_benign_data + train_malignant_data

    val_benign_data = DoubleTreeChannels("../data/"+fold_data+"/val/wave1/benign",
                                                         "../data/"+fold_data+"/val/wave2/benign",
                                                         "../data/"+fold_data+"/val/wave3/benign",
                                                         "../data/"+fold_data+"/val/wave4/benign",
                                                         "benign",
                                                         transform=transform)

    val_malignant_data = DoubleTreeChannels(
        "../data/"+fold_data+"/val/wave1/malignant",
        "../data/"+fold_data+"/val/wave2/malignant",
        "../data/"+fold_data+"/val/wave3/malignant",
        "../data/"+fold_data+"/val/wave4/malignant",
        "malignant",
        transform=transform)

    val_data = val_benign_data + val_malignant_data

    test_benign_data = DoubleTreeChannels("../data/"+fold_data+"/test/wave1/benign",
                                                          "../data/"+fold_data+"/test/wave2/benign",
                                                          "../data/"+fold_data+"/test/wave3/benign",
                                                          "../data/"+fold_data+"/test/wave4/benign",
                                                          "benign",
                                                          transform=transform)

    test_malignant_data = DoubleTreeChannels(
        "../data/"+fold_data+"/test/wave1/malignant",
        "../data/"+fold_data+"/test/wave2/malignant",
        "../data/"+fold_data+"/test/wave3/malignant",
        "../data/"+fold_data+"/test/wave4/malignant",
        "malignant",
        transform=transform)

    test_data = test_benign_data + test_malignant_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 persistent_workers=True,
                                 pin_memory=True
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   persistent_workers=True,
                                   pin_memory=True
                                   )

    test_loader = DataLoader(test_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   persistent_workers=True,
                                   pin_memory=True
                                   )



    #设置encoder
    encoder1 = models.resnet18(pretrained=True)
    for param in encoder1.parameters():
        param.requires_grad = False

    for name, param in encoder1.named_parameters():
        if "layer3" in name:
            param.requires_grad = True
        if "layer4" in name:
            param.requires_grad = True
        if "fc" in name:
            param.requires_grad = True

    encoder1 = nn.Sequential(*list(encoder1.children())[:-2])

    encoder2 = models.resnet18(pretrained=True)
    for param in encoder2.parameters():
        param.requires_grad = False

    for name, param in encoder2.named_parameters():
        if "layer3" in name:
            param.requires_grad = True
        if "layer4" in name:
            param.requires_grad = True
        if "fc" in name:
            param.requires_grad = True

    encoder2 = nn.Sequential(*list(encoder2.children())[:-2])

    model = Model(encoder1,encoder2,num_embeddings, embedding_dim, commitment_cost, decay).to(device)


    criterion = WeightedBinaryCrossEntropyLoss(1.5)
    # criterion = WeightedBinaryCrossEntropyLossWithRegularization(2, 0.01)
    criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, amsgrad=False)
    # scheduler = StepLR(optimizer,10,0.1)
    train_res_recon_error = []
    train_res_perplexity = []

    val_res_recon_error = []
    val_res_perplexity = []

    start_time = time.time()  # 记录训练开始时间
    # writer = SummaryWriter("../Logs")
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
            data1, data2, data3, data4, targets, name = batch

            data_path1 = torch.cat([data1, data3, data1-data3], dim=1)
            data_path2 = torch.cat([data2, data4, data2-data4],dim=1)
            data_path1 = data_path1.to(device)
            data_path2 = data_path2.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            vq_loss1,vq_loss2,data_recon1, data_recon2,perplexity1, perplexity2,classifier_outputs = model(data_path1,data_path2)

            data_variance1 = torch.var(data_path1)
            recon_loss1 = F.mse_loss(data_recon1, data_path1) / data_variance1

            classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)

            data_variance2 = torch.var(data_path2)
            recon_loss2 = F.mse_loss(data_recon2, data_path2) / data_variance2


            total_loss = joint_loss_function(recon_loss1,recon_loss2, vq_loss1,vq_loss2, classifier_loss,
                                             lambda_recon1,lambda_recon2, lambda_vq1,lambda_vq2,lambda_classifier
                                             )
            total_loss.backward()
            optimizer.step()
            # scheduler.step()

            predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
            train_score.append(classifier_outputs.cpu().detach().numpy())
            train_pred.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            total_train_loss += total_loss

        # writer.add_scalar('Loss/Train', total_train_loss, epoch)
        val_score = []
        val_pred = []
        val_targets = []
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data1, data2, data3, data4, targets, name = batch

                data_path1 = torch.cat([data1, data3, data1 - data3], dim=1)
                data_path2 = torch.cat([data2, data4, data2 - data4], dim=1)
                data_path1 = data_path1.to(device)
                data_path2 = data_path2.to(device)
                targets = targets.to(device)

                vq_loss1, vq_loss2, data_recon1, data_recon2, perplexity1, perplexity2, classifier_outputs = model(data_path1, data_path2)

                data_variance1 = torch.var(data_path1)
                recon_loss1 = F.mse_loss(data_recon1, data_path1) / data_variance1

                classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)

                data_variance2 = torch.var(data_path2)
                recon_loss2 = F.mse_loss(data_recon2, data_path2) / data_variance2

                total_loss = joint_loss_function(recon_loss1, recon_loss2, vq_loss1, vq_loss2, classifier_loss,
                                                 lambda_recon1, lambda_recon2, lambda_vq1, lambda_vq2, lambda_classifier
                                                 )

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                val_score.append(classifier_outputs.flatten().cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

                total_val_loss += total_loss

        test_score = []
        test_pred = []
        test_targets = []
        total_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                data1, data2, data3, data4, targets, name = batch

                data_path1 = torch.cat([data1, data3, data1 - data3], dim=1)
                data_path2 = torch.cat([data2, data4, data2 - data4], dim=1)
                data_path1 = data_path1.to(device)
                data_path2 = data_path2.to(device)
                targets = targets.to(device)

                vq_loss1, vq_loss2, data_recon1, data_recon2, perplexity1, perplexity2, classifier_outputs = model(data_path1, data_path2)

                data_variance1 = torch.var(data_path1)
                recon_loss1 = F.mse_loss(data_recon1, data_path1) / data_variance1

                classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)

                data_variance2 = torch.var(data_path2)
                recon_loss2 = F.mse_loss(data_recon2, data_path2) / data_variance2

                total_loss = joint_loss_function(recon_loss1, recon_loss2, vq_loss1, vq_loss2, classifier_loss,
                                                 lambda_recon1, lambda_recon2, lambda_vq1, lambda_vq2, lambda_classifier
                                                 )

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                test_score.append(classifier_outputs.flatten().cpu().numpy())
                test_pred.extend(predicted_labels.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

                total_test_loss += total_loss


        # writer.add_scalar('Loss/Val', total_val_loss, epoch)

        if ((epoch + 1) == 40):
            torch.save(model.state_dict(), "../document/models/CLAHE/data2-{}.pth".format(epoch + 1))
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


    # writer.close()
    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")



