import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

if __name__ == '__main__':
    data1 = pd.read_excel("../data/一期单10(1539例)+二期双10(974例)+二期双15(726例)-3239例_20240511.xlsx",sheet_name="all")
    data2 = pd.read_excel("../document/excels/TSRCNet/一期+二期.xlsx",sheet_name="test")

    data = pd.merge(data2,data1,on="dcm_name",how="left")
    data.to_excel("../document/excels/TSRCNet/data1+data2.xlsx",index=False)
