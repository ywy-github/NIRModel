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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你已经定义好了 Model 类和 Classifier 类
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
    def __init__(self, encoder):
        super(Model, self).__init__()

        self._encoder = encoder
        self.classifier = Classifier(512*7*7, 512, 1)

    def forward(self, x):
        z = self._encoder(x)
        classifier_outputs = self.classifier(z.view(z.size(0), -1))
        return classifier_outputs
if __name__ == '__main__':
    # 加载训练好的模型
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
    model = Model(resnet18).to(device)

    # 加载训练好的权重
    model.load_state_dict(torch.load("../models1/package/resnet18-224-15.pth", map_location=device))

    # 设置为评估模式
    model.eval()

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 使用相同的均值和标准差
    ])

    # test_benign_data = MyData("../data/一期数据/val2/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/一期数据/train/malignant", "malignant", transform=transform)
    test_data = test_malignat_data

    test_loader = DataLoader(test_data,
                             batch_size=16,
                             shuffle=True,
                             num_workers=1,
                             persistent_workers=True,
                             pin_memory=True
                             )

    grayscale_cam_list = []
    cam_img_list = []
    image_list = []
    file_name_list = []

    for i, (imgs_s1, targets, dcm_names) in enumerate(test_loader):
        imgs_s1 = torch.cat([imgs_s1] * 3, dim=1)
        target_layers = [model._encoder[-1][1].conv2]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=imgs_s1)
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
            cam_img = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
            cam_img_list.append(cam_img)
            # enhanced_img_list.append(enhanced_img)

            grayscale_cam_list.append(grayscale_cam[i])

        for i in range(len(image_list)):
            # test_excel = pd.read_excel(test_list, header=None)
            # filename_list = test_excel.iloc[1:, 1].tolist()
            # label_list = test_excel.iloc[1:, 4].tolist()
            #
            # f, ax = plt.subplots(1, 3)
            # f.suptitle('label=' + str(label_list[i]) + ' ' + filename_list[i])

            plt.subplot(1, 3, 1)
            plt.imshow(image_list[i][:,:,0], cmap='gray')
            plt.style.use("classic")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(grayscale_cam_list[i])
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(cam_img_list[i])
            plt.axis('off')
            plt.colorbar()
            plt.savefig("../data/camfig-train-18-train-50/" + file_name_list[i] + ".png")

            plt.show()
            a=1