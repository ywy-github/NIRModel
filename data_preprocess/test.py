import os
import shutil

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from openpyxl import Workbook


if __name__ == "__main__":

    # 读取Excel文件
    excel_file = "../data/info.xlsx"
    df_train = pd.read_excel(excel_file, sheet_name="train")
    df_val = pd.read_excel(excel_file, sheet_name="val")
    df_test = pd.read_excel(excel_file, sheet_name="test")

    # 获取所有常规灯板的图片名字
    all_images = set(df_train['dcm_name']).union(set(df_val['dcm_name']), set(df_test['dcm_name']))

    # 原始文件夹路径
    original_folder_path = "../data/qc前二期双十常规灯板"

    # 遍历qc后二期数据文件夹
    for folder_name in ["train", "val", "test"]:
        folder_path = os.path.join(original_folder_path, folder_name)
        if os.path.exists(folder_path):
            for wave_folder in os.listdir(folder_path):
                wave_folder_path = os.path.join(folder_path, wave_folder)
                if os.path.isdir(wave_folder_path):
                    for label_folder in os.listdir(wave_folder_path):
                        label_folder_path = os.path.join(wave_folder_path, label_folder)
                        if os.path.isdir(label_folder_path):
                            for image_name in os.listdir(label_folder_path):
                                # 检查图片是否为常规灯板的图片，如果不是则删除
                                if image_name not in all_images:
                                    image_path = os.path.join(label_folder_path, image_name)
                                    os.remove(image_path)
                                    print(f"Deleted {image_path}")





