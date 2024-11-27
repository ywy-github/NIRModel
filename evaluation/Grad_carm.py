import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练模型
model = torch.load("../models2/筛查重构/VQ-VAE-筛查重构-200.pth", map_location=device)
model.eval()

# 准备输入图像的预处理函数
transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])
# 定义保存梯度和特征图的钩子函数
gradients = None
activations = None

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]  # 保存梯度

def save_activation(module, input, output):
    global activations
    activations = output  # 保存特征图

# 注册钩子到目标层
target_layer = model.layer4[-1]  # ResNet 最后一层的卷积
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

def generate_grad_cam(image_path, output_path):
    global gradients, activations

    # 加载和预处理图像
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

    # 前向传播
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    output[0, pred_class].backward()

    # 计算 Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # 计算权重
    cam = torch.relu(torch.sum(weights * activations, dim=1)).squeeze()  # 加权求和后激活

    # 归一化
    cam = cam - cam.min()
    cam = cam / cam.max()

    # 调整 Grad-CAM 的大小与原始图像匹配
    cam_resized = resize(cam.unsqueeze(0), image.size[::-1]).squeeze().detach().numpy()

    # 可视化
    plt.imshow(image)
    plt.imshow(cam_resized, alpha=0.5, cmap='jet')  # 添加热图叠加
    plt.axis('off')

    # 保存到文件
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# 调用函数，生成 Grad-CAM 并保存
if __name__ == "__main__":
    input_image = "path/to/your/image.jpg"  # 输入图像路径
    output_image = "path/to/save/grad_cam.jpg"  # 输出图像保存路径
    generate_grad_cam(input_image, output_image)
    print(f"Grad-CAM saved to {output_image}")