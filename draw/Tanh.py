import numpy as np
import matplotlib.pyplot as plt

# 定义Tanh函数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

if __name__ == '__main__':
    # 生成x值，从-10到10，共400个点
    x = np.linspace(-10, 10, 400)

    # 计算每个x对应的Sigmoid值
    y = tanh(x)
    dy = tanh_derivative(x)

    # # 绘制Tanh函数图像
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, color='blue')
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, dy, color='red')
    plt.show()