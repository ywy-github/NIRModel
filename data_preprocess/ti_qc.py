import os
import pandas as pd



if __name__ == '__main__':
    # 读取Excel文件
    excel_path = '../data/一期单10(1539例)+二期双10(947例)+二期双15(726例)-训练测试验证-3212例_20231208.xlsx'  # 替换成你的Excel文件路径
    df = pd.read_excel(excel_path,sheet_name="一期单10训练1259")

    # 文件夹路径
    base_folder = '../data/ti_一期数据/train'  # 替换成你的train文件夹路径

    # 遍历数据框的每一行
    for index, row in df.iterrows():
        dcm_name = row['dcm_name']
        tumor_nature = row['tumor_nature']
        new_qc = row['new_qc']

        # 构建图片路径
        img_path = os.path.join(base_folder, 'benign' if tumor_nature == 0 else 'benign', dcm_name)

        # 检查是否合格
        if new_qc != '合格':
            # 如果不合格，删除图片
            if os.path.exists(img_path):
                os.remove(img_path)

