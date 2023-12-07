import os
import shutil

import pandas as pd

if __name__ == '__main__':
    # 读取Excel文件
    data1 = pd.read_excel("../models/result/全增强.xlsx")
    data2 = pd.read_excel("../models/result/全增强+train.xlsx")
    data = pd.merge(data1,data2,on="dcm_name",how="left")
    data.to_excel("../models/result/test.xlsx",index=False)