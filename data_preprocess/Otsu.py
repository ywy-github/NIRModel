import os

import cv2
import numpy as np
from PIL import Image

def apply_otsu(image):
    image = np.array(image)
    if image is None:
        print("Error: Unable to read the image.")
    else:
        # 大津法阈值化
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    return thresholded


def apply_otsu_to_folder(input_folder, output_folder):
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

            # 应用CLAHE增强
            enhanced_image = apply_otsu(image)
            enhanced_image = Image.fromarray(enhanced_image)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "otsu_" + file_name))

if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/test"
    output_folder = "../data/一期数据/otsu_test"

    # 应用otsu
    apply_otsu_to_folder(input_folder, output_folder)


