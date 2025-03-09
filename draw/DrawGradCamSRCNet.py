import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

from Main1.data_loader import MyData
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


        return  classifier_output


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()

        self._encoder = encoder
        self.classifier = Classifier(512*14*14, 512, 1)

    def forward(self, x):
        z = self._encoder(x)
        classifier_outputs = self.classifier(z.view(z.size(0), -1))
        return classifier_outputs

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载训练好的模型
    model = torch.load("../models2/筛查重构/VQ-VAE-筛查重构-200.pth", map_location=device)
    extendModel = ExtendedModel(model).to(device)
    extendModel.load_state_dict(torch.load('../models2/筛查重构+分类联合学习/筛查重构+分类-89.pth'))
    extendModel.eval()  # 切换到评估模式

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


    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 使用相同的均值和标准差
    ])
    model = Model(resnet18).to(device)
    model.load_state_dict(torch.load("../models1/package/resnet18-18.pth", map_location=device))
    model.eval()

    # test_benign_data = MyData("../data/一期数据/val2/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/val/malignant", "malignant", transform=transform)
    test_data = test_malignat_data

    test_loader = DataLoader(test_data,
                             batch_size=16,
                             shuffle=True,
                             num_workers=1,
                             persistent_workers=True,
                             pin_memory=True
                             )

    grayscale_cam_list1 = []
    grayscale_cam_list2 = []
    cam_img_list1 = []
    cam_img_list2 = []
    image_list = []
    file_name_list = []

    for i, (imgs_s1, targets, dcm_names) in enumerate(test_loader):
        imgs_s1 = torch.cat([imgs_s1] * 3, dim=1)
        target_layers1 = [model._encoder[-1][1].conv2]
        target_layers2 = [extendModel.model._encoder[-1][1].conv2]

        cam1 = GradCAM(model=model, target_layers=target_layers1)
        cam2 = GradCAM(model=extendModel, target_layers=target_layers2)

        grayscale_cam1 = cam1(input_tensor=imgs_s1)
        grayscale_cam2 = cam2(input_tensor=imgs_s1)
        # grayscale_cam = grayscale_cam[0, :]
        # grayscale_cam_list.append(grayscale_cam)
        # imgs = imgs_s1[:, 0, :, :].numpy()
        # enhanced_imgs = imgs_s2.numpy()
        # 处理一个batch
        for i in range(imgs_s1.size(0)):
            img = np.float32(imgs_s1[i]).transpose(1, 2, 0) / 255.0
            # enhanced_img = np.float32(enhanced_imgs[i,]).transpose(1, 2, 0)

            # min_val = img.max()
            # max_val = img.min()
            # img_normal = (img - min_val) / (max_val - min_val)
            file_name_list.append(dcm_names[i].replace('.bmp', ''))
            image_list.append(img)  # 取一个通道出来
            cam_img1 = show_cam_on_image(img, grayscale_cam1[i], use_rgb=True)
            cam_img2 = show_cam_on_image(img, grayscale_cam2[i], use_rgb=True)
            cam_img_list1.append(cam_img1)
            cam_img_list2.append(cam_img2)
            # enhanced_img_list.append(enhanced_img)

            grayscale_cam_list1.append(grayscale_cam1[i])
            grayscale_cam_list2.append(grayscale_cam2[i])

        for i in range(len(image_list)):
            # test_excel = pd.read_excel(test_list, header=None)
            # filename_list = test_excel.iloc[1:, 1].tolist()
            # label_list = test_excel.iloc[1:, 4].tolist()
            #
            # f, ax = plt.subplots(1, 3)
            # f.suptitle('label=' + str(label_list[i]) + ' ' + filename_list[i])

            plt.subplot(1, 3, 1)
            plt.imshow(image_list[i][:, :, 0], cmap='gray')
            plt.style.use("classic")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(cam_img_list1[i], cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(cam_img_list2[i], cmap='gray')
            plt.axis('off')
            plt.savefig("../data/camfig/" + "val/" + file_name_list[i] + ".png")

            plt.show()
            a = 1