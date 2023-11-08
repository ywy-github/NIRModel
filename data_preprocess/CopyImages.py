import os
import shutil
import pandas as pd

if __name__ == '__main__':
    # 读取Excel文件
    excel_file = '../data/一期数据.xlsx'
    df = pd.read_excel(excel_file,sheet_name="val")

    # 原始图片文件夹和目标文件夹路径
    source_folder = '../nir3134'
    output_folder = '../data/一期数据/val'  # 存放所有数据的主文件夹

    # 创建主文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历Excel数据
    for index, row in df.iterrows():
        filename = row['dcm_name']
        label = row['tumor_nature']
        label_folder=''
        if(label==0):
            label_folder='benign'
        else:
            label_folder = 'malignant'
        # 构建文件路径
        source_path = os.path.join(source_folder, filename)
        target_folder = os.path.join(output_folder, label_folder)

        # 创建标签文件夹（如果不存在）
        os.makedirs(target_folder, exist_ok=True)

        target_path = os.path.join(target_folder, filename)

        if os.path.exists(source_path):
            # copy图片文件
            shutil.copy(source_path, target_path)
        else:
            print("filename:"+filename)
