import pandas as pd

if __name__ == '__main__':
    data1 = pd.read_excel("../data/info.xlsx")
    data2 = pd.read_excel("../data/二期双10双15-20个特征-20240301.xlsx")
    data3 = pd.read_excel("../data/二期双10双15-质量合格-1078例-20231220.xlsx",sheet_name="test")

    data = pd.merge(data1,data2,on="dcm_name",how='left')

    data.to_excel("../data/malignant.xlsx",index=False)