import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, path, label,transform=None):
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



if __name__ =='__main__':
    root_dir = "../data/train"
    benign_label_dir = "benign"
    malignant_label_dir = "malignant"
    benign_dataset = MyData("../data/train/benign","benign")
    malignat_dataset = MyData("../data/train/malignant","malignant")
    train_dataset = benign_dataset + malignat_dataset
    print(len(train_dataset))

