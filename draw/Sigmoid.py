import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

if __name__ == '__main__':
    # 生成x值，从-10到10，共400个点
    x = np.linspace(-10, 10, 400)

    # 计算每个x对应的Sigmoid值
    y = sigmoid(x)
    dy = sigmoid_derivative(x)

    # # 绘制Sigmoid函数图像
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, color='blue')
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, dy, color='red')
    plt.show()