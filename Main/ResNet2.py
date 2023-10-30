import time

import torch
from matplotlib import pyplot as plt
from tensorboard import summary
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from Main.Metrics import all_metrics
from Main.data_loader import MyData

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 100
    learning_rate = 1e-5

    # 读取数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    train_benign_data = MyData("../data/一期数据/train/benign", "benign", transform=transform)
    train_malignat_data = MyData("../data/一期数据/train/malignant", "malignant", transform=transform)
    train_data = train_benign_data + train_malignat_data

    val_benign_data = MyData("../data/一期数据/val/benign", "benign", transform=transform)
    val_malignat_data = MyData("../data/一期数据/val/malignant", "malignant", transform=transform)
    val_data = val_benign_data + val_malignat_data

    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    validation_loader = DataLoader(val_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)



    model = models.resnet18(pretrained=False)

    #调整结构
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_classes = 1
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.add_module("sigmoid", nn.Sigmoid())

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_time = time.time()  # 记录训练开始时间
    train_losses = []
    validation_losses = []
    for epoch in range(epochs):
        model.train()
        train_predictions = []
        train_targets = []
        total_train_loss = 0
        for batch in training_loader:
            images, targets, names= batch
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted_labels = (output >= 0.5).int().squeeze()
            train_predictions.extend(predicted_labels.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_losses.append(total_train_loss)

        model.eval()
        val_predictions = []
        val_targets = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                images, targets, names = batch
                images = images.to(device)
                targets = targets.to(device)
                output = model(images)
                loss = criterion(output, targets)

                total_val_loss += loss.item()
                predicted_labels = (output >= 0.5).int().squeeze()
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                validation_losses.append(total_val_loss)

        if ((epoch + 1) % 1 == 0):
            # torch.save(models, "VQ_VAE{}.pth".format(i+1))
            print('%d epoch' % (epoch + 1))

            train_acc, train_sen, train_spe = all_metrics(train_targets, train_predictions)
            print("训练集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
                  "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(total_train_loss))

            val_acc, val_sen, val_spe = all_metrics(val_targets, val_predictions)
            print("验证集 acc: {:.4f}".format(val_acc) + "sen: {:.4f}".format(val_sen) +
                  "spe: {:.4f}".format(val_spe) + "loss: {:.4f}".format(total_val_loss))

    # 结束训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")

    plt.figure(figsize=(10, 5))
    plt.plot(range(10, epochs + 1, 10), train_losses, label='Train Loss')
    plt.plot(range(10, epochs + 1, 10), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.show()