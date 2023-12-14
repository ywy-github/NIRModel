import os

import cv2
import numpy as np
from PIL import Image

def adaptive_gaussian_filter(image, max_window_size=7, sigma=1, threshold=10):
    """
    自适应高斯滤波器。

    Parameters:
    - image: PIL Image 对象或 NumPy 数组，表示输入图像。
    - max_window_size: 高斯滤波器的最大窗口大小。
    - sigma: 高斯分布的标准差。
    - threshold: 自适应调整的阈值。

    Returns:
    - filtered_image: PIL Image 对象，表示滤波后的图像。
    """

    # 将 PIL Image 转换为 NumPy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 获取图像梯度
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算像素梯度幅度
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 计算局部方差
    local_var = cv2.GaussianBlur(gradient_magnitude**2, (max_window_size, max_window_size), sigma) - \
                cv2.GaussianBlur(gradient_magnitude, (max_window_size, max_window_size), sigma)**2

    # 初始化窗口大小，设置为奇数
    window_size = (3, 3)

    # 根据局部方差自适应调整窗口大小
    window_size = tuple(int(s) if int(s) % 2 != 0 else int(s) + 1 for s in window_size)  # 确保是奇数
    window_size = np.clip(np.asarray(window_size), 3, max_window_size)  # 限制窗口大小在合理范围内

    # 自适应高斯滤波
    filtered_image = cv2.GaussianBlur(image, tuple(window_size), sigma)

    # 将 NumPy 数组转换为 PIL Image
    filtered_image = Image.fromarray(filtered_image.astype(np.uint8))

    return filtered_image


def apply_gaussian_to_folder(input_folder, output_folder):
    # 创建保存目录
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入目录中的每个子目录（benign和malignant）
    for class_folder in os.listdir(input_folder):
        class_input_path = os.path.join(input_folder, class_folder)
        class_output_path = os.path.join(output_folder, class_folder)

        # 创建保存增强图像的子目录
        os.makedirs(class_output_path, exist_ok=True)

        # 遍历每个子目录中的图像文件
        for file_name in os.listdir(class_input_path):
            file_path = os.path.join(class_input_path, file_name)
            output_file_path = os.path.join(class_output_path, file_name)

            # 读取图像
            image = Image.open(file_path)

            enhanced_image = adaptive_gaussian_filter(image)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "gauss_" + file_name))

if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/test"
    output_folder = "../data/一期gauss/test"

    # 应用gamma增强
    apply_gaussian_to_folder(input_folder, output_folder)
