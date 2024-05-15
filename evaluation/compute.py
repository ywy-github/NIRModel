import numpy as np

if __name__ == '__main__':

    values = [0.6882,0.6882,0.7200,0.7010,0.7129]

    # 计算均值
    mean = sum(values) / len(values)

    # 计算标准差
    std = np.std(values)

    print("均值（平均数）:", mean)
    print("标准差:", std)