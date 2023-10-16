import os
import shutil
import pandas as pd

# 读取Excel文件
excel_file = '../data/test.xlsx'
df = pd.read_excel(excel_file,sheet_name="测试集710例")

# 原始图片文件夹和目标文件夹路径
source_folder = '../data2'
output_folder = '../data/test'  # 存放所有数据的主文件夹

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

    # 移动图片文件
    shutil.move(source_path, target_path)
