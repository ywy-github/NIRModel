from __future__ import print_function

import matplotlib.pyplot as plt
from six.moves import xrange

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)
lowlight_images_path = 'E:/ImgData/hunheData/'
test_images_path = 'E:/ImgData/finall_data/testdata/test'


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.*")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = np.expand_dims(data_lowlight, axis=2)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        # data_lowlight = transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))

        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


train_dataset = lowlight_loader(lowlight_images_path)
#
test_dataset = lowlight_loader(test_images_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):#生成K个D维向量
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim # D
        self._num_embeddings = num_embeddings #K

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
    def __init__(self, in_channel, num_hidden, num_residual_hiddens, ch=1):
        super(Residual, self).__init__()
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=num_residual_hiddens,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_residual_hiddens,
                               out_channels=num_hidden,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        if ch != 1:
            self.downsample = nn.Conv2d(in_channels=in_channel, out_channels=num_hidden, kernel_size=(1, 1),
                                        stride=1)
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        out = self.conv1(self.relu1(x))
        out = self.bn1(out)

        out = self.conv2(self.relu2(out))
        out = self.bn2(out)
        identity = self.downsample(x)

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
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)


    def forward(self, inputs):

        x = self._conv_1(inputs)
        x = self._residual_stack(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()



        self._residual_stack = ResidualStack(in_channels=in_channels,
                                             num_hiddens=in_channels,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)


    def forward(self, inputs):

        x = self._residual_stack(inputs)
        x = self._conv_trans_1(x)
        return x

batch_size = 64
training_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(test_dataset,
                               batch_size=16,
                               shuffle=True,
                               pin_memory=True)


def generate_and_save_images(predictions, epoch):
    # predictions = model.sample(test_input)
    predictions = predictions.permute(0, 3, 2, 1)
    predictions = predictions.data.cpu().numpy()
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('img_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close()




num_training_updates = 1500

num_hiddens = 8
num_residual_hiddens = 8
num_residual_layers = 2

embedding_dim = 8
num_embeddings = 64

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._res_block1 = Residual(in_channel=1,num_hidden=num_hiddens,num_residual_hiddens=num_residual_hiddens,ch=0)

        self._encoder1 = Encoder(num_hiddens, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._encoder1_1 = Encoder(num_hiddens, num_hiddens,
                                 num_residual_layers,
                                 num_residual_hiddens)

        self._encoder2 = Encoder(num_hiddens, 2*num_hiddens,
                                 num_residual_layers,
                                 2*num_residual_hiddens)
        self._encoder2_1 = Encoder( 2*num_hiddens, 4 * num_hiddens,
                                 num_residual_layers,
                                 4* num_residual_hiddens)

        self._encoder3 = Encoder(4*num_hiddens,8* num_hiddens,
                                 num_residual_layers,
                                 8* num_residual_hiddens)
        self._encoder3_1 = Encoder(8 * num_hiddens, 16 * num_hiddens,
                                 num_residual_layers,
                                 16 * num_residual_hiddens)
        self._res_block2 = Residual(in_channel=16 * num_hiddens, num_hidden=16*num_hiddens, num_residual_hiddens=16*num_hiddens)


        self._pre_vq_conv1 = nn.Conv2d(in_channels=2*num_hiddens+8,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._pre_vq_conv2 = nn.Conv2d(in_channels= 8* num_hiddens,
                                      out_channels=8*embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._pre_vq_conv3 = nn.Conv2d(in_channels=16*num_hiddens,
                                      out_channels=16*embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae1 = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
            self._vq_vae2 = VectorQuantizerEMA(8*num_embeddings, 8*embedding_dim,
                                              commitment_cost, decay)
            self._vq_vae3 = VectorQuantizerEMA(16*num_embeddings, 16*embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae1 = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
            self._vq_vae2 = VectorQuantizer(8*num_embeddings, 8*embedding_dim,
                                           commitment_cost)
            self._vq_vae3 = VectorQuantizer(16*num_embeddings, 16*embedding_dim,
                                           commitment_cost)



        self._decoder3 = Decoder(16*embedding_dim,
                                8*embedding_dim,
                                num_residual_layers,
                                8*num_residual_hiddens)
        self._decoder3_1 = Decoder(8*embedding_dim,
                                 4*embedding_dim,
                                 num_residual_layers,
                                 4*num_residual_hiddens)

        self._res_block3 = Residual(in_channel= 8*num_hiddens, num_hidden=8*num_hiddens, num_residual_hiddens=8*num_residual_hiddens)
        self._decoder2 = Decoder(8*embedding_dim,
                                4*embedding_dim,
                                num_residual_layers,
                                2*num_residual_hiddens)
        self._decoder2_1 = Decoder(4*embedding_dim,
                                 2*embedding_dim,
                                 num_residual_layers,
                                 num_residual_hiddens)
        self._res_block4 = Residual(in_channel= 2*num_hiddens+8, num_hidden=2*num_hiddens+8, num_residual_hiddens=2*num_residual_hiddens+8)

        self._decoder1 = Decoder(embedding_dim,
                                 num_hiddens,
                                 num_residual_layers,
                                 num_residual_hiddens)
        self._decoder1_1 = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._res_block5 = Residual(num_hiddens, 1, num_residual_hiddens=num_residual_hiddens,ch=0)

    def forward(self, x):
        re1 = self._res_block1(x)
        en1 = self._encoder1(re1)
        en1_1 = self._encoder1_1(en1)
        en2 = self._encoder2(en1_1)
        en2_1 = self._encoder2_1(en2)
        en3 = self._encoder3(en2_1)
        en3_1 = self._encoder3_1(en3)

        vq3 = self._res_block2(en3_1)
        vq3 = self._pre_vq_conv3(vq3)
        loss3, quantized3, perplexity3, _  = self._vq_vae3(vq3)

        de3 = self._decoder3(quantized3)
        de3_1 = self._decoder3_1(de3)

        de3_1 = torch.cat((de3_1,en2_1),dim=1)
        vq2 = self._res_block3(de3_1)
        vq2 = self._pre_vq_conv2(vq2)
        loss2, quantized2, perplexity2, _ = self._vq_vae2(vq2)

        de2 = self._decoder2(quantized2)
        de2_1 = self._decoder2_1(de2)

        de2_1 = torch.cat((de2_1, en1_1), dim=1)
        vq1= self._res_block4(de2_1)
        vq1 = self._pre_vq_conv1(vq1)
        loss1, quantized1, perplexity1, _ = self._vq_vae1(vq1)

        de1 = self._decoder1(quantized1)
        x_recon = self._decoder1_1(de1)
        x_recon = self._res_block5(x_recon)

        return loss1+loss2+loss3, x_recon, perplexity1+perplexity2+perplexity3


model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []

for i in xrange(num_training_updates):
    for index_num, data in enumerate(training_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # generate_and_save_images(all_data, i)

        vq_loss, data_recon, perplexity = model(data)
        data_variance = torch.var(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        # if index_num % 100 == 0:
        #     print(index_num)
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % 10 == 0:
        generate_and_save_images(data_recon, i)
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()
