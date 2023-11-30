import os
import cv2
import numpy as np
from PIL import Image

def apply_clahe(image, clip_limit=3.0, grid_size=(8, 8)):
    if image.mode == 'L':
        gray_image = np.array(image)
    else:
        # 转换为灰度图像
        gray_image = np.array(image.convert("L"))

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # 应用CLAHE
    enhanced_image = clahe.apply(gray_image)

    # 将CLAHE增强的图像转换回PIL格式
    enhanced_image_pil = Image.fromarray(enhanced_image)

    return enhanced_image_pil

def apply_clahe_to_folder(input_folder, output_folder):
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
            enhanced_image = apply_clahe(image)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "clahe_" + file_name))

if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/二期双十/train"
    output_folder = "../data/二期双十/new_train"

    # 应用CLAHE增强
    apply_clahe_to_folder(input_folder, output_folder)
