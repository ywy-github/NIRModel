import os
import shutil

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from openpyxl import Workbook


if __name__ == "__main__":

    # 设置文件夹路径

    source_dir = '../data/二期双10双15第二波段原始图'  # 原始图片所在文件夹
    nir1_dir = '../data/二期双十+双十五/train/wave1/malignant'  # NIR1图片所在文件夹
    target_dir = '../data/二期双十+双十五/train/wave4/malignant'  # 目标文件夹，复制图片到这里

    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

        # 遍历原始文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 检查文件是否为图片（这里假设所有文件都是图片，或者您可以添加检查逻辑）
        if os.path.isfile(os.path.join(source_dir, filename)):
            # 检查NIR1文件夹中是否有同名文件
            if os.path.exists(os.path.join(nir1_dir, filename)):
                # 构造源文件和目标文件的完整路径
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, filename)

                # 复制文件
                shutil.copy2(source_file, target_file)  # copy2会尝试保留文件的元数据
                print(f"Copied {source_file} to {target_file}")
            else:
                print(f"No matching file found in {nir1_dir} for {filename}")

    print("All matching files have been copied.")





