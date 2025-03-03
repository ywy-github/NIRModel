import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
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

        return loss, x_recon, perplexity, classifier_outputs

class ExtendedModel(nn.Module):
    def __init__(self, model):
        super(ExtendedModel, self).__init__()
        self.model = model
        self.classifier = Classifier(512*14*14, 512,1)

    def forward(self, x):
        z = self.model._encoder(x)
        loss, quantized, perplexity, _ = self.model._vq_vae(z)
        classifier_output = self.classifier(quantized.view(quantized.size(0),-1))
        x_recon = self.model._decoder(quantized)


        return loss, x_recon, perplexity, classifier_output

def tsne_visualization(model, dataloader, num_samples=500, perplexity=30.0, save_path="tsne_visualization.png"):
    """
    对模型提取的特征使用 t-SNE 降维并进行三维可视化。

    Args:
        model: 已训练好的模型。
        dataloader: 数据加载器。
        num_samples: 用于可视化的样本数量。
        perplexity: t-SNE 的 perplexity 参数。
        save_path: 保存图像的路径。

    Returns:
        None
    """
    features = []
    labels = []

    # 提取特征
    with torch.no_grad():
        for batch in dataloader:
            data, targets, _ = batch
            data = torch.cat([data] * 3, dim=1).to(device)  # 扩展到 3 通道
            targets = targets.to(device)

            z = model.model._encoder(data)  # 使用模型的编码器部分
            loss, quantized, perplexity_tensor, _ = model.model._vq_vae(z)
            quantized = quantized.view(quantized.size(0), -1)  # 展平张量

            # 将特征移到CPU并转换为numpy
            features.append(quantized.cpu().numpy())  # quantized 在 CPU 上进行处理
            labels.extend(targets.cpu().numpy())  # 标签也移动到CPU

            # 如果达到样本数量限制则提前退出
            if len(labels) >= num_samples:
                break

    # 转换为 NumPy 数组
    features = np.vstack(features)[:num_samples]
    labels = np.array(labels)[:num_samples]

    # 确保 perplexity 是一个浮动类型的数值
    perplexity = float(perplexity)

    # 使用 t-SNE 降维为 3 维
    tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=100, n_iter=2000, random_state=42)
    features_3d = tsne.fit_transform(features)

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})

    # 绘制三维散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    # 根据实际情况指定每个标签对应的颜色，例如：红色恶性，蓝色良性
    color_map = {0: 'blue', 1: 'red'}  # 假设标签为0是良性，1是恶性

    # 使用不同颜色绘制不同类别的点
    for label in unique_labels:
        indices = labels == label
        ax.scatter(features_3d[indices, 0], features_3d[indices, 1], features_3d[indices, 2],
                   label=f'Class {int(label)}', color=color_map[label], s=30)

    # 设置标题和坐标轴标签
    ax.set_title('t-SNE Visualization of Extracted Features (3D)', fontsize=20)
    ax.legend(fontsize=20)

    # 保存图片
    # plt.savefig(save_path, dpi=300)

    # 显示图片
    plt.show()

if __name__ == '__main__':
    # 假设已经加载了训练好的模型
    # extendModel 是已经训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../models2/筛查重构/VQ-VAE-筛查重构-200.pth", map_location=device)
    extendModel = ExtendedModel(model).to(device)
    extendModel.load_state_dict(torch.load('../models2/筛查重构+分类联合学习/筛查重构+分类-89.pth'))
    extendModel.eval()  # 切换到评估模式

    batch_size = 16
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    test_benign_data = MyData("../data/一期数据/train_2/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/train_2/malignant", "malignant", transform=transform)
    test_data = test_benign_data + test_malignat_data

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=6,
                             persistent_workers=True,
                             pin_memory=True
                             )
    # 定义 t-SNE 的降维函数
    # save_path = "D:\\PyCharm 2022.2.1\\workspace\\NIRModel\\picture\\model1.png"
    # 调用函数进行可视化
    tsne_visualization(extendModel, test_loader, num_samples=200, perplexity=30)
