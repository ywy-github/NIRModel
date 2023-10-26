import time

import numpy as np
import torch
from torchvision import transforms
# from models import *
# 准备数据集
from torch import optim
from torch.utils.data import DataLoader
from Main.Metrics import all_metrics
from Main.models.Vq_VAE_Join_Classifier_multi_route import Model, Focal_Loss, joint_loss_function
from Main.data_loader import MyData
import torch.nn.functional as F

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 500

    num_hiddens = 64
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = num_hiddens * 3
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    weight_positive = 2  # 调整这个权重以提高对灵敏度的重视

    learning_rate = 1e-5

    lambda_recon = 0.2
    lambda_vq = 0.2
    lambda_classifier = 0.6

    # 读取数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/NIR_Wave2/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/NIR_Wave2/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/NIR_Wave2/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/NIR_Wave2/val/malignant", "malignant", transform=transform)
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

    criterion = Focal_Loss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    # scheduler = StepLR(optimizer,50,0.1)
    train_res_recon_error = []
    train_res_perplexity = []

    val_res_recon_error = []
    val_res_perplexity = []
    total_train_loss = []
    total_val_loss = []
    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        train_predictions = []
        train_targets = []

        for batch in training_loader:
            data, targets, dcm_names = batch
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity, classifier_outputs = model(data)

            data_variance = torch.var(data)
            recon_loss = F.mse_loss(data_recon, data) / data_variance
            classifier_loss = criterion(classifier_outputs, targets.view(-1, 1))
            total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                             lambda_classifier)
            total_loss.backward()
            optimizer.step()
            # scheduler.step()

            predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
            train_predictions.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            train_res_recon_error.append(recon_loss.item())
            train_res_perplexity.append(perplexity.item())
            total_train_loss.append(total_loss.item())
        val_predictions = []
        val_targets = []

        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                data, targets, names = batch
                data = data.to(device)
                targets = targets.to(device)
                vq_loss, data_recon, perplexity, classifier_outputs = model(data)
                data_variance = torch.var(data)
                recon_loss = F.mse_loss(data_recon, data) / data_variance
                classifier_loss = criterion(classifier_outputs, targets.view(-1, 1))
                total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                                 lambda_classifier)

                predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                val_res_recon_error.append(recon_loss.item())
                val_res_perplexity.append(perplexity.item())
                total_val_loss.append(total_loss.item())
        # 将测试步骤中的真实数据、重构数据和上述生成的新数据绘图

        if ((epoch + 1) % 30 == 0):
            torch.save(model, "../models/VQ_VAE_Join_Classifier/{}.pth".format(epoch + 1))
            # concat = torch.cat((all_data[0].view(128, 128),
            #                     data_recon[0].view(128, 128)), 1)
            # plt.matshow(concat.cpu().detach().numpy())
            # plt.show()

            print('%d iterations' % (epoch + 1))
            train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
            print("训练集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
                  "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(np.mean(total_train_loss[-10:])))

            val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
            print("验证集 acc: {:.4f}".format(val_acc) + "sen: {:.4f}".format(val_sen) +
                  "spe: {:.4f}".format(val_spe) + "loss: {:.4f}".format(np.mean(total_val_loss[-10:])))

            print('train_recon_error: %.3f' % np.mean(train_res_recon_error[-10:]))
            print('train_perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
            print('val_recon_error: %.3f' % np.mean(val_res_recon_error[-10:]))
            print('val_perplexity: %.3f' % np.mean(val_res_perplexity[-10:]))

    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")


