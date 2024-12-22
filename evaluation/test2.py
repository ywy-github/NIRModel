import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


# 遍历文件夹并收集文件名
def collect_image_filenames(folder_path):
    filenames = set()
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):  # 支持常见的图片格式
                filenames.add(file)
    return filenames


# 对比两个文件夹中相同的图片文件名
def compare_folders_by_name(folder1, folder2):
    # 获取两个文件夹下的图片文件名
    filenames_folder1 = collect_image_filenames(folder1)
    filenames_folder2 = collect_image_filenames(folder2)

    # 找出相同的文件名
    common_filenames = filenames_folder1 & filenames_folder2

    # 输出相同的图片文件名
    print(f"共有 {len(common_filenames)} 张图片在两个文件夹中是相同的。")
    for filename in common_filenames:
        print(f"相同图片：{filename}")


# 使用示例



if __name__ == '__main__':
    folder1 = "../data/一期数据/train"
    folder2 = "../data/一期数据/test"
    compare_folders_by_name(folder1, folder2)