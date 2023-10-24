import pandas as pd
import numpy as np

if __name__ == '__main__':
    # data_2202 = pd.read_excel("F:\\PyCharm 2022.2.1\\pythonProject\\nir\\总体数据2202+932.xlsx")
    data = pd.read_excel("F:\\PyCharm 2022.2.1\\pythonProject\\nir\\总体数据2202+932.xlsx",sheet_name="测试集932例")

    # 创建一个布尔掩码以筛选包含特定关键词的行
    keyword = "单波段"
    mask = data['dcm_name'].str.contains(keyword)

    # 使用掩码来提取符合条件的行
    filtered_data = data[mask]
    filtered_data.to_excel("../all_data/single.xlsx",index=False)