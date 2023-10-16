from __future__ import print_function

import time

import matplotlib.pyplot as plt
from six.moves import xrange

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm

from Main.Metrics import all_metrics
from Main.data_loader import MyData

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


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens,strides=1):
        super(Residual, self).__init__()
        self.relu1 = nn.LeakyReLU(inplace=True)
        self. conv1 = nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hiddens)
        if strides != 1:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=1,
                                        stride=strides, bias=False)
        else:
            self.downsample = None  # Set to None when strides == 1

    def forward(self, x):
        out = self.conv1(self.relu1(x))
        out = self.bn1(out)

        out = self.conv2(self.relu2(out))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        return out + identity


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

def generate_and_save_images(predictions, epoch):
  # predictions = model.sample(test_input)
  predictions = predictions.permute(0,3,2,1)
  predictions = predictions.data.cpu().numpy()
  fig = plt.figure(figsize=(4,4))
  for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(predictions[i,:,:,0],cmap='gray')
    plt.axis('off')
  plt.savefig('img_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()
  plt.close()

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self.classifier = Classifier(8*32*32,256,2)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        recon_loss, quantized, perplexity, _ = self._vq_vae(z)
        classifier_outputs = self.classifier(quantized)
        x_recon = self._decoder(quantized)

        return recon_loss, x_recon, perplexity, classifier_outputs

# 定义联合模型的损失函数
def joint_loss_function(recon_loss,vq_loss,classifier_outputs,targets,lambda_recon,lambda_vq,lambda_classifier):
    # 分类器损失
    classifier_loss = nn.CrossEntropyLoss()(classifier_outputs, targets)

    # 总损失
    total_loss = lambda_recon * recon_loss + lambda_vq*vq_loss + lambda_classifier * classifier_loss

    return total_loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 1000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 8
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-5

    lambda_recon = 0.4
    lambda_vq =  0.2
    lambda_classifier = 0.4

    # 读取数据集
    transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])

    train_benign_data = MyData("../data/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    validation_loader = DataLoader(val_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    train_res_recon_error = []
    train_res_perplexity = []

    val_res_recon_error = []
    val_res_perplexity = []

    start_time = time.time()  # 记录训练开始时间

    for epoch in xrange(epochs):
        model.train()
        train_predictions = []
        train_targets = []
        total_train_loss = 0
        for batch in training_loader:
            data, targets = batch
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity,classifier_outputs = model(data)

            data_variance = torch.var(data)
            recon_loss = F.mse_loss(data_recon, data) / data_variance
            total_loss = joint_loss_function(recon_loss,vq_loss,classifier_outputs,targets,lambda_recon,lambda_vq,lambda_classifier)
            total_loss.backward()
            optimizer.step()

            _, predicted = torch.max(classifier_outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            train_res_recon_error.append(recon_loss.item())
            train_res_perplexity.append(perplexity.item())
            total_train_loss += total_loss.item()
        val_predictions = []
        val_targets = []
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data, targets = batch
                data = data.to(device)
                targets = targets.to(device)
                vq_loss, data_recon, perplexity, classifier_outputs = model(data)
                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                total_loss = joint_loss_function(recon_loss,vq_loss,classifier_outputs,targets,lambda_recon,lambda_vq,lambda_classifier)

                _, predicted = torch.max(classifier_outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                val_res_recon_error.append(recon_loss.item())
                val_res_perplexity.append(perplexity.item())
                total_val_loss +=total_loss.item()
        # 将测试步骤中的真实数据、重构数据和上述生成的新数据绘图

        if ((epoch + 1) % 100 == 0):
            torch.save(model, "VQ_VAE_join_classifier{}.pth".format(epoch + 1))
            concat = torch.cat((data[0].view(128, 128),
                                data_recon[0].view(128, 128)), 1)
            plt.matshow(concat.cpu().detach().numpy())
            plt.show()

            print('%d iterations' % (epoch + 1))
            train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
            print("训练集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
                  "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(total_train_loss))

            val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
            print("验证集 acc: {:.4f}".format(val_acc) + "sen: {:.4f}".format(val_sen) +
                  "spe: {:.4f}".format(val_spe) + "loss: {:.4f}".format(total_val_loss))

            print('train_recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('train_perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print('val_recon_error: %.3f' % np.mean(val_res_recon_error[-100:]))
            print('val_perplexity: %.3f' % np.mean(val_res_perplexity[-100:]))

    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")
