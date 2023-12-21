import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

if __name__ == '__main__':

    image = Image.open("../data/一期数据")


    img_array = (data * 255).astype(np.uint8)

    # 遍历每个张量中的图像，保存为灰度图像
    for i in range(img_array.shape[0]):
        # 选择一个通道，比如 R、G 或 B，这里选择第一个通道
        grayscale_img = img_array[i, 0, :, :]  # 假设第一个通道是灰度通道

        # 创建 PIL Image 对象
        pil_image = Image.fromarray(grayscale_img)

        # 保存图像
        pil_image.save(f'../data/out/image_{i}.png')