import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from Main.Metrics import all_metrics
from Main.data_loader import MyData

test_benign_data = MyData("../data/test/benign/images", "benign")
test_malignat_data = MyData("../data/test/malignant/images", "malignant")
test_data = test_benign_data + test_malignat_data
print("测试数据集的长度为：{}".format(len(test_data)))

# 利用 DataLoader 来加载数据集
test_dataloader = DataLoader(test_data, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8192, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 这里将在GPU上训练的模型，拿到CPU上进行测试，需要添加map_location
model = torch.load("tudui_0.pth", map_location=torch.device('cpu'))
test_predictions = []
test_targets = []

model.eval()
with torch.no_grad():
    for data in test_dataloader:
        images, targets = data
        output = model(images)
        _, predicted = torch.max(output, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

    test_acc, test_sen, test_spe = all_metrics(test_targets, test_predictions)
    print("测试集上的准确率: {:.4f}".format(test_acc))
    print("测试集上的准确率: {:.4f}".format(test_sen))
    print("测试集上的准确率: {:.4f}".format(test_spe))
