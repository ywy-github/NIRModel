import pandas as pd

if __name__ == '__main__':
    data1 = pd.read_excel("../data/resnet18-双路径-增-增-相减-原-原-相减.xlsx",sheet_name = "test")
    data2 = pd.read_excel("../data/一期单10(1539例)+二期双10(947例)+二期双15(726例)-训练测试验证-3212例_20231208.xlsx",sheet_name = "二期双10测试192")
    # data3 = pd.read_excel("../data/二期双10双15-质量合格-1078例-20231220.xlsx",sheet_name="test")

    data = pd.merge(data1,data2,on="dcm_name",how='left')

    data.to_excel("../data/qc后样本判断情况.xlsx",index=False)