import numpy as np
import matplotlib.pyplot as plt

# 定义ReLu函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
if __name__ == '__main__':
    # 生成x值，从-10到10，共400个点
    x = np.linspace(-10, 10, 400)

    # 计算每个x对应的Sigmoid值
    y = relu(x)
    dy = relu_derivative(x)

    # 绘制ReLu函数图像
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue')
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(x, dy, color='red')
    # plt.show()