import os
import shutil

import pandas as pd
from openpyxl.workbook import Workbook

if __name__ == '__main__':
     data_all = pd.read_excel("../data/总体数据2202+932.xlsx",sheet_name="测试集932例")
     data = pd.read_excel("../data/一期数据.xlsx",sheet_name="测试集275例")

     data1 = pd.merge(data,data_all,how="left",on="dcm_name")
     data1.to_excel("../data/test.xlsx",index=False)

