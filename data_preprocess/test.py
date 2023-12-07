import os
import shutil

import pandas as pd

if __name__ == '__main__':
    # 读取Excel文件
    excel_file_path = '../data/省肿瘤病理dcm-20231206(1).xlsx'
    df = pd.read_excel(excel_file_path)

    # 文件夹路径
    source_folder = '../data/省肿瘤'
    benign_folder = '../data/省肿瘤/benign'
    malignant_folder = '../data/省肿瘤/malignant'

    # 遍历Excel表格中的每一行
    for index, row in df.iterrows():
        dcm_name = row['dcm_name']
        tumor_nature = row['tumor_nature']

        # 构建源文件路径和目标文件夹路径
        source_file_path = os.path.join(source_folder, dcm_name)
        target_folder = benign_folder if tumor_nature == 0 else malignant_folder

        # 如果目标文件夹不存在，则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 构建目标文件路径
        target_file_path = os.path.join(target_folder, dcm_name)

        # 移动文件
        shutil.move(source_file_path, target_file_path)

    print("文件移动完成。")