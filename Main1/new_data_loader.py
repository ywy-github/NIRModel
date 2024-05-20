import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, path, label , age_column, cup_size_column, transform=None):
        self.path = path
        self.transform = transform
        self.label = label
        # 读取包含标签、年龄和罩杯信息的Excel文件
        self.df = pd.read_excel(os.path.join(path, 'info.xlsx'))

        # 图像文件夹下的所有图像文件名
        self.image_path_list = os.listdir(os.path.join(path, 'images').replace("\\", "/"))

        # 列名
        self.age_column = age_column
        self.cup_size_column = cup_size_column

        # 类别映射
        self.label_mapping = {'benign': 0, 'benign': 1}

    def __getitem__(self, idx):
        name = self.image_path_list[idx]
        img = Image.open(os.path.join(self.path, 'images', name))
        if self.transform:
            img = self.transform(img)

        label = self.label_mapping[self.label]
        age = float(self.df.loc[idx, self.age_column])
        cup_size = self.df.loc[idx, self.cup_size_column]

        return img, label, age, cup_size, name

    def __len__(self):
        return len(self.image_path_list)

if __name__ == '__main__':
    # 在info.xlsx文件中的列名
    age_column = 'age'
    cup_size_column = 'cup_size'

    benign_dataset = MyData("../data/一期age_cupsize/train/benign", "benign",age_column,cup_size_column)
    malignat_dataset = MyData("../data/一期age_cupsize/train/malignant", "benign",age_column,cup_size_column)
    train_dataset = benign_dataset + malignat_dataset
