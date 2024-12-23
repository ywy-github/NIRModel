import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = []
        self.gradients = []

        # 注册钩子
        self.hook()

    def hook(self):
        # 将目标层的特征图保存到self.feature_maps中
        def save_features(module, input, output):
            self.feature_maps.append(output)

        # 将梯度保存到self.gradients中
        def save_gradients(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        # 注册钩子
        self.target_layer.register_forward_hook(save_features)
        self.target_layer.register_backward_hook(save_gradients)

    def generate_cam(self, input_image, target_class=None):
        # 1. 执行前向传播
        input_image = input_image.unsqueeze(0).to(device)
        self.model.zero_grad()

        # 获取模型输出
        output = self.model(input_image)

        # 如果没有指定目标类别，则选择预测类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        # 2. 执行反向传播
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        # 3. 获取梯度和特征图
        gradients = self.gradients[0]
        feature_maps = self.feature_maps[0]

        # 4. 计算每个特征图的权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1).squeeze()

        # 5. 进行 ReLU 处理，确保热图中不会有负值
        cam = F.relu(cam)

        # 6. 对热图进行归一化处理
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        return cam

    def visualize_cam(self, cam, input_image, save_path=None):
        # 将生成的 Grad-CAM 转换为 numpy 格式，并将其映射到原始图像大小
        cam = cam.cpu().detach().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_image.size(2), input_image.size(1)))  # Resize to input image size

        # 将原始图像转换为 NumPy 数组并归一化
        img = input_image.squeeze().cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))  # 转换为 HWC 格式
        img = np.uint8((img * 255))  # 转换为 0-255 的图像

        # 将热图与原始图像叠加
        heatmap = np.uint8(255 * cam)  # 将热图范围设置为 0-255
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用 Jet 色彩映射
        superimposed_img = heatmap * 0.4 + img  # 叠加热图

        # 可视化图像
        plt.imshow(superimposed_img)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    # 假设你选择的目标层是模型的最后一个卷积层
    target_layer = model._encoder.b3[1].conv2  # 选择你想用来做 Grad-CAM 的卷积层

    # 创建 GradCAM 对象
    grad_cam = GradCAM(model, target_layer)

    # 准备一个输入图像并将其传递给 Grad-CAM
    input_image = Image.open('path_to_input_image.jpg')
    input_image = transform(input_image).to(device)  # 转换为模型的输入格式

    # 生成 Grad-CAM
    cam = grad_cam.generate_cam(input_image)

    # 可视化 Grad-CAM
    grad_cam.visualize_cam(cam, input_image)
