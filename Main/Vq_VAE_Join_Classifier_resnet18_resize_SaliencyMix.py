from __future__ import print_function

import time

import cv2
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

from Metrics import all_metrics
from data_loader import MyData

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

        self.classifier = Classifier(512*14*14,512,1)

        self._decoder = Decoder()

    def forward(self, x):
        z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        classifier_outputs = self.classifier(quantized.view(quantized.size(0),-1))
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity, classifier_outputs

# 定义联合模型的损失函数
def joint_loss_function(recon_loss,vq_loss,classifier_loss,lambda_recon,lambda_vq,lambda_classifier):
    # 总损失
    total_loss = lambda_recon * recon_loss + lambda_vq*vq_loss + lambda_classifier * classifier_loss

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

def saliencyMix(x, y, beta=1.0):
    batch_size = x.size(0)

    lam = np.random.beta(beta, beta)

    # 生成随机的索引排列，用于选择混合的另一个样本
    index = torch.randperm(batch_size)

    # 生成随机的矩形框坐标
    bbx1, bby1, bbx2, bby2 = saliency_bbox(x.size(), lam)

    # 将另一张图像的一部分替换到当前图像中
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    # 计算混合后的标签
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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

    lambda_recon = 0.2
    lambda_vq = 0.2
    lambda_classifier = 0.6

    beta = 1
    salmix_prob = 0.8

    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/ti_一期数据/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/ti_一期数据/val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data


    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 pin_memory=True
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   persistent_workers=True,
                                   pin_memory=True
                                  )


    #设置encoder
    encoder = models.resnet18(pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False

    for name, param in encoder.named_parameters():
        if "layer3" in name:
            param.requires_grad = True
        if "layer4" in name:
            param.requires_grad = True
        if "fc" in name:
            param.requires_grad = True

    encoder = nn.Sequential(*list(encoder.children())[:-2])

    model = Model(encoder,num_embeddings, embedding_dim, commitment_cost, decay).to(device)


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
            data, targets, dcm_names = batch
            data = torch.cat([data] * 3, dim=1)
            data = data.to(device)
            targets = targets.to(device)

            r = np.random.rand(1)
            if beta > 0 and r < salmix_prob:
                # generate mixed sample
                data, target_a, target_b, lam = saliencyMix(data, targets, beta)
                # compute output

                vq_loss, data_recon, perplexity, classifier_outputs = model(data)

                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                classifier_loss = lam * criterion(target_a.view(-1, 1), classifier_outputs) + (1 - lam) * criterion(
                    target_b.view(-1, 1), classifier_outputs)
                total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                                 lambda_classifier)
            else:
                vq_loss, data_recon, perplexity, classifier_outputs = model(data)

                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)
                total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                                 lambda_classifier)

            total_loss.backward()
            optimizer.step()
            # scheduler.step()

            predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
            train_score.append(classifier_outputs.cpu().detach().numpy())
            train_pred.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            total_train_loss += total_loss
            train_res_recon_error.append(recon_loss.item())
            train_res_perplexity.append(perplexity.item())
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
                vq_loss, data_recon, perplexity, classifier_outputs = model(data)
                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)
                total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                                 lambda_classifier)

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                val_score.append(classifier_outputs.flatten().cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

                total_val_loss += total_loss
                val_res_recon_error.append(recon_loss.item())
                val_res_perplexity.append(perplexity.item())
        # writer.add_scalar('Loss/Val', total_val_loss, epoch)
        #
        # if ((epoch + 1) == 63 or (epoch + 1) == 65):
        #     torch.save(model.state_dict(), "../models/qc/VQ-VAE-resnet18-qc-二期双十-{}.pth".format(epoch + 1))
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

        print('train_recon_error: %.3f' % np.mean(train_res_recon_error[-10:]))
        print('train_perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
        print('val_recon_error: %.3f' % np.mean(val_res_recon_error[-10:]))
        print('val_perplexity: %.3f' % np.mean(val_res_perplexity[-10:]))

    # writer.close()
    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")



