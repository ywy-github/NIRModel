import os

import pandas as pd
from PIL.Image import Image
from d2l import torch
from matplotlib import transforms
from torch.utils.data import DataLoader, Dataset


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("../models/result/VQ-VAE-resnet18_data1.pth", map_location=device)

    img = Image.open("../data/一期数据/test/benign/010-BJBA-00001-DL-201707280918-D.bmp")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])
    img = transform(img)

    vq_loss, data_recon, perplexity, classifier_outputs = model(img)

    print(classifier_outputs)
