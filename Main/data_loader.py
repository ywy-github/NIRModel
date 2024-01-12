import numpy as np
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
        self.image_path_list1 = os.listdir(self.path1)
        self.image_path_list2 = os.listdir(self.path2)
        self.label_mapping = {'benign': 0, 'malignant': 1}

    def __getitem__(self, idx):
        name1 = self.image_path_list1[idx]
        name2 = self.image_path_list2[idx]
        self.image_path1 = os.path.join(self.path1, name1)
        self.image_path2 = os.path.join(self.path2, name2)
        img1 = Image.open(self.image_path1)
        img2 = Image.open(self.image_path2)
        img = img1-img2
        if self.transform:
            img1 = self.transform(img1)
            imag2 = self.transform(img2)
            img = self.transform(img)
        label = self.label_mapping[self.label]
        return img1, img2, img, label, name1

    def __len__(self):
        return len(self.image_path_list)




if __name__ =='__main__':

    benign_dataset = MyData("../data/train/benign","benign")
    malignat_dataset = MyData("../data/train/malignant","malignant")
    train_dataset = benign_dataset + malignat_dataset
    print(len(train_dataset))

