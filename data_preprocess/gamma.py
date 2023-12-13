import os
import cv2
import numpy as np
from PIL import Image

def apply_gamma(image, gamma):
    # 第一次伽马校正
    output = np.power(image, gamma)
    # # 第二次伽马校正
    # output = np.power(temp, gamma)

    # 将图像的像素值归一化到 0 到 255 的范围
    output = ((output - np.min(output)) / (np.max(output) - np.min(output)) * 255).astype(np.uint8)

    # 将NumPy数组转换为Image对象
    enhanced_image = Image.fromarray(output)

    return enhanced_image

def apply_gamma_to_folder(input_folder, output_folder):
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

            # 应用gamma增强
            enhanced_image = apply_gamma(image, 1.5)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "gamma_" + file_name))

if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/train"
    output_folder = "../data/一期gamma/train"

    # 应用gamma增强
    apply_gamma_to_folder(input_folder, output_folder)
