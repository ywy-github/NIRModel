from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np



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

        self.classifier = Classifier(200704, 512, 1)

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

def pred(NIR_path,origin_img1_path,enhanced_img2_path,origin_img2_path,work_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = 64
    num_embeddings = 512  # 和encoder输出维度相同，和decoder输入维度相同

    commitment_cost = 0.25

    decay = 0.99
    encoder1 = models.resnet18(pretrained=False)
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

    encoder2 = models.resnet18(pretrained=False)
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

    model = Model(encoder1, encoder2, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    model_path = work_dir + '一期+二期-100.pth'
    model.load_state_dict(torch.load(model_path,map_location=device))

    # 读取四种图片，分别是第一波段增强图，第一波段原始图，第二波段增强图，第二波段原始图
    image1 = Image.open(NIR_path)
    image2 = Image.open(origin_img1_path)
    image3 = Image.open(enhanced_img2_path)
    image4 = Image.open(origin_img2_path)

    transform = transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    image1 = transform(image1).unsqueeze(0).to(device)
    image2 = transform(image2).unsqueeze(0).to(device)
    image3 = transform(image3).unsqueeze(0).to(device)
    image4 = transform(image4).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()  # 确保模型处于评估模式
        data_path1 = torch.cat([image1,image3,image1 - image3], dim=1)
        data_path2 = torch.cat([image2, image4, image2 - image4], dim=1)
        _,_,_, _,_, _,classifier_outputs = model(data_path1,data_path2)
        # print("Predicted Probability:", "{:.6f}".format(classifier_outputs.item()))
    return classifier_outputs.detach().cpu().numpy().item()


if __name__ == '__main__':
    image_name = "020-ZSSYX-00305-CWME-202203240949-双波段-L-D.bmp"
    # 读取四种图片，分别是第一波段增强图，第一波段原始图，第二波段增强图，第二波段原始图

    NIR_path = "../data/一期+二期/test/wave1/benign/" + image_name
    origin_img1_path = "../data/一期+二期/test/wave2/benign/" + image_name
    enhanced_img2_path = "../data/一期+二期/test/wave3/benign/" + image_name
    origin_img2_path = "../data/一期+二期/test/wave4/benign/" + image_name

    work_dir = "../models1/package/"

    classifier_outputs = pred(NIR_path,origin_img1_path,enhanced_img2_path,origin_img2_path,work_dir)
    print("Predicted Probability:", "{:.6f}".format(classifier_outputs))