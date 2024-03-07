import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, path, label, transform=None):
        self.path = path
        self.label = label
        self.transform = transform
        self.image_path_list = os.listdir(self.path)
        self.label_mapping = {'benign': 0, 'malignant': 1}

    def __getitem__(self, idx):
        name = self.image_path_list[idx]
        self.image_path = os.path.join(self.path, name)
        img = Image.open(self.image_path)
        if self.transform:
            img = self.transform(img)
        label = self.label_mapping[self.label]
        return img, label, name

    def __len__(self):
        return len(self.image_path_list)


class TreeChannels(Dataset):
    def __init__(self, path1, path2, label, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.label = label
        self.transform = transform
        self.image_path_list = os.listdir(self.path1)
        self.label_mapping = {'benign': 0, 'malignant': 1}

    def __getitem__(self, idx):
        name = self.image_path_list[idx]

        self.image_path1 = os.path.join(self.path1, name)
        self.image_path2 = os.path.join(self.path2, name)
        img1 = Image.open(self.image_path1)
        img2 = Image.open(self.image_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = self.label_mapping[self.label]
        return img1, img2, label, name

    def __len__(self):
        return len(self.image_path_list)


class DoubleTreeChannels(Dataset):
    def __init__(self, path1, path2, path3, path4, label, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4
        self.label = label
        self.transform = transform
        self.image_path_list = os.listdir(self.path1)
        self.label_mapping = {'benign': 0, 'malignant': 1}

    def __getitem__(self, idx):
        name = self.image_path_list[idx]

        self.image_path1 = os.path.join(self.path1, name)
        self.image_path2 = os.path.join(self.path2, name)
        self.image_path3 = os.path.join(self.path3, name)
        self.image_path4 = os.path.join(self.path4, name)
        img1 = Image.open(self.image_path1)
        img2 = Image.open(self.image_path2)
        img3 = Image.open(self.image_path3)
        img4 = Image.open(self.image_path4)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        label = self.label_mapping[self.label]
        return img1, img2, img3, img4, label, name

    def __len__(self):
        return len(self.image_path_list)

class DoubleTreeChannelsOtherInformation(Dataset):
    def __init__(self, path1, path2, path3, path4, excel,label, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4
        self.label = label
        self.transform = transform
        self.image_path_list = os.listdir(self.path1)
        self.label_mapping = {'benign': 0, 'malignant': 1}
        # 读取包含标签、年龄和罩杯信息的Excel文件
        self.df = pd.read_excel(excel)

    def __getitem__(self, idx):
        name = self.image_path_list[idx]

        self.image_path1 = os.path.join(self.path1, name)
        self.image_path2 = os.path.join(self.path2, name)
        self.image_path3 = os.path.join(self.path3, name)
        self.image_path4 = os.path.join(self.path4, name)
        img1 = Image.open(self.image_path1)
        img2 = Image.open(self.image_path2)
        img3 = Image.open(self.image_path3)
        img4 = Image.open(self.image_path4)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        label = self.label_mapping[self.label]
        #辅助信息
        age = float(self.df.loc[idx,'age'])
        cup_size = self.df.loc[idx, 'cup_size']
        H_lso3 = self.df.loc[idx, ' H_lso3']
        dnirs_L1max = float(self.df.loc[idx,'dnirs_L1max'])
        H_Bsc1 = float(self.df.loc[idx, 'H_Bsc1'])
        dnirs_L1min = float(self.df.loc[idx, 'dnirs_L1min'])
        std_HomH_L0_L1 = float(self.df.loc[idx, 'std_HomH_L0_L1'])
        HistDiffH_L1_6 = float(self.df.loc[idx, 'HistDiffH-L1_6'])
        L1_L0_Q31 = float(self.df.loc[idx, 'L1-L0_Q31'])
        dnirs_Hmin = float(self.df.loc[idx, 'dnirs_Hmin'])
        HistDiffL0_L1_9 = float(self.df.loc[idx, 'HistDiffL0-L1_9'])
        HistDiffH_L1_4 = float(self.df.loc[idx, 'HistDiffH-L1_4'])
        HskewH_L1 = float(self.df.loc[idx, 'HskewH-L1'])
        dnirs_Hstd = float(self.df.loc[idx, 'dnirs_Hstd'])
        dnirs_Hmed = float(self.df.loc[idx, 'dnirs_Hmed'])
        dnirs_L1med = float(self.df.loc[idx, 'dnirs_L1med'])
        L0_Nvm_std = float(self.df.loc[idx, 'L0_Nvm_std'])
        meanL0_L1 = float(self.df.loc[idx, 'meanL0-L1'])
        L1_Nvm_std = float(self.df.loc[idx, 'L1_Nvm_std'])
        H_Bsc4 = float(self.df.loc[idx, 'H_Bsc4'])
        L1_L0_MedQ1 = float(self.df.loc[idx, 'L1-L0_MedQ1'])
        L1_L0_VArea = float(self.df.loc[idx, 'L1-L0_VArea'])

        return img1, img2, img3, img4, label, name,age,cup_size,H_lso3,dnirs_L1max,H_Bsc1,dnirs_L1min,std_HomH_L0_L1,\
               HistDiffH_L1_6,L1_L0_Q31,dnirs_Hmin,HistDiffL0_L1_9,HistDiffH_L1_4,HskewH_L1,dnirs_Hstd,dnirs_Hmed,\
               dnirs_L1med,L0_Nvm_std,meanL0_L1,L1_Nvm_std,H_Bsc4,L1_L0_MedQ1,L1_L0_VArea


    def __len__(self):
        return len(self.image_path_list)


class DoubleTreeChannelsNoLabel(Dataset):
    def __init__(self, path1, path2, path3, path4, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4
        self.transform = transform
        self.image_path_list = os.listdir(self.path1)
    def __getitem__(self, idx):
        name = self.image_path_list[idx]

        self.image_path1 = os.path.join(self.path1, name)
        self.image_path2 = os.path.join(self.path2, name)
        self.image_path3 = os.path.join(self.path3, name)
        self.image_path4 = os.path.join(self.path4, name)
        img1 = Image.open(self.image_path1)
        img2 = Image.open(self.image_path2)
        img3 = Image.open(self.image_path3)
        img4 = Image.open(self.image_path4)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        return img1, img2, img3, img4, name

    def __len__(self):
        return len(self.image_path_list)



if __name__ =='__main__':

    train_benign_data_wave1 = TreeChannels("../data/ti_二期双十+双十五wave1/train/benign", "../data/ti_二期双十+双十五wave2原始图/train/benign" ,"benign")
    train_malignat_data_wave1 = TreeChannels("../data/ti_二期双十+双十五wave1/train/malignant","../data/ti_二期双十+双十五wave2原始图/train/malignant"  ,"malignant")
    train_data_wave1 = train_benign_data_wave1 + train_malignat_data_wave1

