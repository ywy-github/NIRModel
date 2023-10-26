import os
import shutil

import pandas as pd
from openpyxl.workbook import Workbook

if __name__ == '__main__':

    # 设置源文件夹和目标文件夹的路径
    source_dir = "F:\\PyCharm 2022.2.1\\pythonProject\\nir\\NIR-wave"
    target_wave1_dir = "../data/wave1"
    target_wave2_dir = "../data/wave2"

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_wave1_dir):
        os.makedirs(target_wave1_dir)
    if not os.path.exists(target_wave2_dir):
        os.makedirs(target_wave2_dir)

    # 定义源文件夹中包含图片的文件夹列表
    source_folders = ["benign-0", "benign-1", "malignant-0", "malignant-1"]

    # 遍历源文件夹和文件夹列表，然后将图片复制到目标文件夹
    for folder in source_folders:
        source_wave1_path = os.path.join(source_dir, "test", folder)
        source_wave2_path = os.path.join(source_dir, "test", folder.replace('-0', '-1'))

        if os.path.exists(source_wave1_path):
            for filename in os.listdir(source_wave1_path):
                source_image_path = os.path.join(source_wave1_path, filename)
                target_image_path = os.path.join(target_wave1_dir, filename)
                shutil.copy(source_image_path, target_image_path)

        if os.path.exists(source_wave2_path):
            for filename in os.listdir(source_wave2_path):
                source_image_path = os.path.join(source_wave2_path, filename)
                target_image_path = os.path.join(target_wave2_dir, filename)
                shutil.copy(source_image_path, target_image_path)



