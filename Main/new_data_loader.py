import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, path,excel_path,sheet_name,label_column, age_column, cup_size_column, transform=None):
        self.path = path
        self.transform = transform

        # 读取包含标签、年龄和罩杯信息的Excel文件
        self.df = pd.read_excel(excel_path,sheet_name=sheet_name)

        # 图像文件夹下的所有图像文件名
        self.image_path_list = os.listdir(path)

        # 列名
        self.label_column = label_column
        self.age_column = age_column
        self.cup_size_column = cup_size_column

        # 类别映射
        self.label_mapping = {'benign': 0, 'malignant': 1}

    def __getitem__(self, idx):
        name = self.image_path_list[idx]
        img = Image.open(os.path.join(self.path, 'images', name))
        if self.transform:
            img = self.transform(img)

        label = self.label_mapping[self.df.loc[idx, self.label_column]]
        age = float(self.df.loc[idx, self.age_column])
        cup_size = self.df.loc[idx, self.cup_size_column]

        return img, label, age, cup_size, name

    def __len__(self):
        return len(self.image_path_list)

if __name__ == '__main__':
    excel_path = "../data/一期单10(1539例)+二期双10(947例)+二期双15(726例)-训练测试验证-3212例_20231208.xlsx"
    sheet_name = "一期单10训练1259"
    root_dir = "../data/一期数据/train"
    benign_label_dir = "benign"
    malignant_label_dir = "malignant"

    # 在info.xlsx文件中的列名
    label_column = 'tumor_nature'
    age_column = 'age'
    cup_size_column = 'cup_size'

    custom_dataset = MyData(root_dir,excel_path,sheet_name,label_column, age_column, cup_size_column)
    print(len(custom_dataset))
