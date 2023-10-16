import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from Main.Metrics import all_metrics
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

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VAEAndClassifier(nn.Module):
    def __init__(self, vae, classifier):
        super(VAEAndClassifier, self).__init__()
        self.vae = vae
        self.classifier = classifier

    def forward(self, x):
        mu, log_var = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, log_var)
        recon_x = self.vae.decoder(z)
        classifier_outputs = self.classifier(z)
        return recon_x, mu, log_var, classifier_outputs

# 创建模型和优化器

latent_dim = 32
encoder = Encoder(1, latent_dim)
decoder = Decoder(latent_dim, 1)

# 创建联合模型
vae = VAE(encoder, decoder)
classifier = Classifier(latent_dim, 16, 2)
joint_model = VAEAndClassifier(vae, classifier)


# 定义联合模型的损失函数
def joint_loss_function(recon_x, x, mu, log_var, classifier_outputs, targets, lambda_recon=0.2, lambda_classifier=0.3):
    # VAE重构损失
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # VAE KL散度损失
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # 分类器损失
    classifier_loss = nn.CrossEntropyLoss()(classifier_outputs, targets)

    # 总损失
    total_loss = lambda_recon * recon_loss + lambda_classifier * classifier_loss + kl_loss

    return total_loss


# 定义优化器
optimizer = optim.Adam(joint_model.parameters(), lr=0.001)

#读取数据集
transform = transforms.Compose([transforms.Resize([128,128],transforms.ToTensor())])

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
# 训练联合模型
num_epochs = 10
for epoch in range(num_epochs):
    joint_model.train()
    train_predictions = []
    train_targets = []
    total_train_loss = 0.0
    for batch in train_dataloader:
        x, targets = batch
        if torch.cuda.is_available():
            x = x.cuda()
            targets = targets.cuda()
        recon_x, mu, log_var, classifier_outputs = joint_model(x)
        loss = joint_loss_function(recon_x, x, mu, log_var, classifier_outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(classifier_outputs, 1)
        train_predictions.extend(predicted.cpu().numpy())
        train_targets.extend(targets.cpu().numpy())
        total_train_loss += loss.item()

    train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
    print("训练集 acc: {:.4f}".format(train_acc)+"sen: {:.4f}".format(train_sen)+
          "spe: {:.4f}".format(train_spe)+"loss: {:.4f}".format(total_train_loss))


    val_predictions = []
    val_targets = []
    total_val_loss = 0.0
    vae.eval()
    with torch.no_grad():
        for data in val_dataloader:
            x, targets = batch
            if torch.cuda.is_available():
                x = x.cuda()
                targets = targets.cuda()
            recon_x, mu, log_var, classifier_outputs = joint_model(x)
            loss = joint_loss_function(recon_x, x, mu, log_var, classifier_outputs, targets)

            _, predicted = torch.max(classifier_outputs, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
            total_val_loss += loss.item()

        val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
        print("验证集 acc: {:.4f}".format(val_acc) + "sen: {:.4f}".format(val_sen) +
              "spe: {:.4f}".format(val_spe) + "loss: {:.4f}".format(total_val_loss))
    torch.save(joint_model, "joint_model_{}.pth".format(epoch))