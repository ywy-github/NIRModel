import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image



if __name__ == '__main__':

    data1 = pd.read_excel("../data/一期数据.xlsx", sheet_name="val")
    data2 = pd.read_excel("../data/111.xlsx")

    # 使用 left 和 right 参数
    data = pd.merge(left=data1, right=data2, how="left", on="dcm_name")

    data.to_excel("../data/test.xlsx", index=False)
