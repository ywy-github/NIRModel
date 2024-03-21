import os
import pandas as pd


if __name__ == '__main__':
    # 读取Excel文件
    excel_file = "../data/二期双十剔除年龄大于60岁/test/exclude大于60岁.xlsx"
    df = pd.read_excel(excel_file)

    # 获取dcm_name列，并转换为list
    dcm_names = df['dcm_name'].tolist()

    # 指定文件夹路径
    folder_path = "../data/二期双十剔除年龄大于60岁/test/wave4"

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件名，假设文件名格式为"xxx.dcm"
            file_name = file

            # 检查文件名是否在列表中，并且对应样本年龄大于60岁
            if file_name in dcm_names:
                print("1")
                # 移除文件
                os.remove(file_path)

