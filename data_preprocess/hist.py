import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # 打开图像
    image_path = '../data/一期数据/new_train/malignant/clahe_021-SHZL-00022-XML-201708091549-D.bmp'
    image = Image.open(image_path)

    # 将Pillow Image对象转换为NumPy数组
    image_np = np.array(image)

    # 使用Pillow库提供的histogram方法计算直方图
    hist = image.histogram()

    # 绘制直方图
    plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.title('Pillow Histogram')

    # 调整纵轴比例
    plt.yscale('log')  # 或者 'linear'，'log'，'symlog' 等其他选项

    plt.show()

    # 使用Pillow库提供的方法展示图像
    image.show()
