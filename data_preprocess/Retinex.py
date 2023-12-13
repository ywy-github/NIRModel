import os

import cv2
import numpy as np
from PIL import Image


def single_scale_retinex(image, sigma):
    # 使用高斯滤波平滑图像
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # 计算图像的对数域表示
    log_image = np.log1p(image.astype(float))
    log_blurred = np.log1p(blurred.astype(float))

    # 计算反射分量
    reflection = log_image - log_blurred

    # 将反射分量映射回0-255范围
    reflection = (reflection - np.min(reflection)) / (np.max(reflection) - np.min(reflection)) * 255

    return reflection


def multi_scale_retinex(image, scales):
    result = np.zeros_like(image, dtype=float)

    for scale in scales:
        # 对每个尺度应用单尺度Retinex
        reflection = single_scale_retinex(image, scale)

        # 将反射分量叠加到结果中
        result += reflection

    # 将结果映射回0-255范围
    result = (result / len(scales)).astype(np.uint8)

    return result
def apply_retinex_to_folder(input_folder, output_folder):
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

            scales = [5, 50, 250]

            # 应用GLCM增强
            enhanced_image = multi_scale_retinex(image,scales)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "retinex_" + file_name))


if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/test"
    output_folder = "../data/一期Retinex/test"

    # 应用增强
    apply_retinex_to_folder(input_folder, output_folder)
