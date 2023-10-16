import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from Main.data_loader import MyData


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 32, 32)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')   #重建误差BCE是二进制交叉熵损失，用于衡量重构数据与原始数据的差异
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())   #KL散度KLD用于衡量潜在表示的分布与标准正态分布之间的差异
    return BCE + KLD

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备
# 创建模型和优化器
latent_dim = 20
encoder = Encoder(1, latent_dim)
decoder = Decoder(latent_dim, 1)
vae = VAE(encoder, decoder)
if torch.cuda.is_available():
    vae = vae.cuda()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

#读取数据集
train_benign_data = MyData("../data/train/benign","benign",transform=transform)
train_malignat_data = MyData("../data/train/malignant","malignant",transform=transform)
train_data = train_benign_data + train_malignat_data


val_benign_data = MyData("../data/val/benign","benign",transform=transform)
val_malignat_data = MyData("../data/val/malignant","malignant",transform=transform)
val_data = val_benign_data + val_malignat_data

# 创建数据加载器
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=64)
val_dataloader = DataLoader(val_data, batch_size=64)
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    total_train_loss = 0
    for batch in train_dataloader:
        imgs, targets = batch
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        recon_x, mu, log_var = vae(imgs)
        loss = loss_function(recon_x, imgs, mu, log_var)
        if torch.cuda.is_available():
            loss = loss.cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {total_train_loss}')
    torch.save(vae, "vae_{}.pth".format(epoch))

    vae.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in val_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            recon_x, mu, log_var = vae(imgs)
            loss = loss_function(recon_x, imgs, mu, log_var)
            total_val_loss += loss.item()

    sample = torch.randn(1,latent_dim).to(device)
    #用decoder生成新数据
    gen = vae.decoder(sample)[0].view(128,128)
    #将测试步骤中的真实数据、重构数据和上述生成的新数据绘图
    concat = torch.cat((imgs[0].view(128, 128),
            recon_x[0].view(128,128), gen), 1)
    plt.matshow(concat.cpu().detach().numpy())
    plt.show()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {total_val_loss}')