import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_preprocess.DobiImage import DubiImage

if __name__ == '__main__':

    fold_path = "../data/省肿瘤493dcm"
    for file_name in os.listdir(fold_path):
        # if file_name == "029-XAJD-00115-YXF-201807091046-D.dcm" or\
        #         file_name == "0571-ZKYZL-S376-ZEHU-202310101624-双波段10-R.dcm" or\
        #          file_name == "0571-ZKYZL-S367-SSFE-202309221041-双波段10-L.dcm" or\
        #          file_name == "0571-ZKYZL-S149-XCJU-202209191631-双波段15-R.dcm":
        #    continue
        file_path = os.path.join(fold_path, file_name)
        images = DubiImage(file_path)
        print(f"{file_name}")
        mainImage1,num_light = images.getMainImgae2AndNum_light()

        if mainImage1 is not None:
            # 指定保存图片的文件夹路径
            save_folder = "../data/省肿瘤493wave2原始图"

            # 确保保存文件夹存在，如果不存在则创建
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 将所有图像进行累加
            total_sum = np.sum(mainImage1, axis=0)

            # 计算前num_light张图的累加值
            first_n_sum = np.sum(mainImage1[:num_light], axis=0)

            if num_light == 3:
                # 计算结果
                result = total_sum - 23 * first_n_sum

            elif num_light == 5:
                result = total_sum - 13 * first_n_sum

            else: print("没有这个灯")

            # 将 result 映射到 [0, 255] 范围，并转换为 uint8 类型
            result = (result - result.min()) / (result.max() - result.min()) * 255
            result = result.astype(np.uint8)

            # 将 result 保存为 BMP 图像
            result_image = Image.fromarray(result)

            bmp_file_name = os.path.splitext(file_name)[0] + ".bmp"

            bmp_file_path = os.path.join(save_folder, bmp_file_name)
            result_image.save(bmp_file_path)

