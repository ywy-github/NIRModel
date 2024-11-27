from __future__ import print_function
import math
import time
import basicblock as B
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MSFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MSFEblock, self).__init__()
        out_channels = in_channels

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),#groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)
    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)
        #res = torch.cat([y0,y1,y2,y3], dim=1)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att+x)

class SKConv(nn.Module):
    def __init__(self, features, M: int = 2, G: int = 32, r: int = 16, stride: int = 1, L: int = 32):
        super().__init__()
        d = int(max(features / r, L))
        self.M = M
        self.features = features
        # 1.split
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        # 2.fuse
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 3.select
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        batch_size = x.shape[0]
        # 1.split
        feats = torch.cat((x, y), dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        # 2.fuse
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        # 3.select
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)

        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V

class DS(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.point_conv = nn.Conv2d(input_channels,2*input_channels,1,1)
        self.Bn = nn.BatchNorm2d(2*input_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.point_conv(x)
        x = self.Bn(x)
        x = self.relu(x)
        return x

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
        act = self.sigmoid(avg_out + max_out)
        out = x * act
        return out

    #空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv1(x1)
        x1 = self.sigmoid(x1)
        out = x * x1
        return out

    #先接通道注意力模块、再接空间注意力模块
class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x*self.ca(x)
        result = out*self.sa(out)
        return x+result

# class MSBlock(nn.Module):
#     def __init__(self, in_nc=3, nc=64, nb=3):  ##act_mode='L'3;2  128
#         super(MSBlock, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc, mode='C')
#         m_body = [MSFEblock(nc, [2, 4, 8]) for _ in range(nb)]
#         self.m_body = nn.Sequential(B.ShortcutBlock(B.sequential(*m_body)))
#         self.m_tail = B.conv(nc, nc, mode='C')
#
#     def forward(self, x):
#         x1 = self.m_head(x)
#         x2 = self.m_body(x1)
#         SR_feature = self.m_tail(x2)
#         return SR_feature

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride=stride)
        self.bn1 = norm_layer(width)
        self.conv2 = MSFEblock(width, atrous_rates=[1, 2, 3])
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.CBAM1 = CBAM(256)
        self.CBAM2 = CBAM(512)
        self.CBAM3 = CBAM(1024)
        self.CBAM4 = CBAM(2048)

        self.DS1 = DS(256)
        self.DS2 = DS(512)
        self.DS3 = DS(1024)
        self.DS4 = DS(2048)

        self.SKConv1 = SKConv(512)
        self.SKConv2 = SKConv(1024)
        self.SKConv3 = SKConv(2048)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层，输入图像尺寸减半
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        m1 = self.CBAM1(x1)
        f1 = self.DS1(m1)

        x2 = self.layer2(x1)
        m2 = self.CBAM2(x2)
        sk1 = self.SKConv1(f1, m2)
        f2 = self.DS2(sk1)

        x3 = self.layer3(x2)
        m3 = self.CBAM3(x3)
        sk2 = self.SKConv2(f2, m3)
        f3 = self.DS3(sk2)

        x4 = self.layer4(x3)
        m4 = self.CBAM4(x4)
        out = self.SKConv3(f3, m4)

        return out


class ResNet50(ResNet):
    def __init__(self, num_classes=2, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        # 调用父类 ResNet 的构造函数，使用 Bottleneck 作为 block，并固定层配置为 [3, 4, 6, 3]
        super(ResNet50, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )


class MSFSSLModel(nn.Module):
    def __init__(self,encoder,num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(MSFSSLModel, self).__init__()

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

        self.classifier = Classifier(2048*14*14,2048,1)

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

    batch_size = 2
    epochs = 1000

    embedding_dim = 64
    num_embeddings = 512  # 和encoder输出维度相同，和decoder输入维度相同

    commitment_cost = 0.25

    decay = 0.99


    learning_rate = 1e-5

    lambda_recon = 0.2
    lambda_vq = 0.2
    lambda_classifier = 0.6

    # 读取数据集
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train/malignant", "benign", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/一期数据/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/一期数据/val/malignant", "benign", transform=transform)
    val_data = val_benign_data + val_malignat_data

    test_benign_data = MyData("../data/一期数据/test/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/test/malignant", "benign", transform=transform)
    test_data = test_benign_data + test_malignat_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1
                                 )

    validation_loader = DataLoader(val_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1
                                  )

    test_loader = DataLoader(test_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1
                                   )


    #设置encoder
    encoder = ResNet50(num_classes=2)
    # for param in encoder.parameters():
    #     param.requires_grad = False
    #
    # for name, param in encoder.named_parameters():
    #     if "layer3" in name:
    #         param.requires_grad = True
    #     if "layer4" in name:
    #         param.requires_grad = True
    #     if "fc" in name:
    #         param.requires_grad = True
    #
    # encoder = nn.Sequential(*list(encoder.children())[:-2])

    MSFSSLModel = MSFSSLModel(encoder, num_embeddings, embedding_dim, commitment_cost, decay).to(device)


    criterion = WeightedBinaryCrossEntropyLoss(2)
    # criterion = WeightedBinaryCrossEntropyLossWithRegularization(2, 0.01)
    criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, MSFSSLModel.parameters()), lr=learning_rate, amsgrad=False)
    # scheduler = StepLR(optimizer,10,0.1)
    train_res_recon_error = []
    train_res_perplexity = []

    val_res_recon_error = []
    val_res_perplexity = []

    test_res_recon_error = []
    test_res_perplexity = []

    start_time = time.time()  # 记录训练开始时间
    # writer = SummaryWriter("../Logs")
    for epoch in range(epochs):
        # if((epoch+1) % 10 == 0):
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.8
        MSFSSLModel.train()
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

            vq_loss, data_recon, perplexity, classifier_outputs = MSFSSLModel(data)

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
        MSFSSLModel.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                vq_loss, data_recon, perplexity, classifier_outputs = MSFSSLModel(data)
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

        test_score = []
        test_pred = []
        test_targets = []
        total_test_loss = 0.0
        MSFSSLModel.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, targets, names = batch
                data = torch.cat([data] * 3, dim=1)
                data = data.to(device)
                targets = targets.to(device)
                vq_loss, data_recon, perplexity, classifier_outputs = MSFSSLModel(data)
                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                classifier_loss = criterion(targets.view(-1, 1), classifier_outputs)
                total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                                 lambda_classifier)

                predicted_labels = (classifier_outputs >= 0.5).int().view(-1)
                test_score.append(classifier_outputs.flatten().cpu().numpy())
                test_pred.extend(predicted_labels.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

                total_test_loss += total_loss
                test_res_recon_error.append(recon_loss.item())
                test_res_perplexity.append(perplexity.item())
        # writer.add_scalar('Loss/Val', total_val_loss, epoch)

        # if ((epoch + 1) == 61 or (epoch + 1) == 65 or (epoch + 1) == 67):
        #     torch.save(model.state_dict(),"../models1/qc/VQ-VAE-resnet18-qc-二期双十+双十五-{}.pth".format(epoch + 1))
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

        print('train_recon_error: %.3f' % np.mean(train_res_recon_error[-10:]))
        print('train_perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
        print('val_recon_error: %.3f' % np.mean(val_res_recon_error[-10:]))
        print('val_perplexity: %.3f' % np.mean(val_res_perplexity[-10:]))

    # writer.close()
    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")



