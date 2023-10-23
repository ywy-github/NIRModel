import os
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix
from skimage import img_as_ubyte

# 指定文件夹的路径
folder_path = '../data/train/benign'  # 替换为您的文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 打开图像文件
    original_image = Image.open(file_path)

    # 将图像复制到一个NumPy数组，并将其数据类型转换为可写的
    original_image_array = np.array(original_image, dtype=np.uint8)

    # 选择GLCM参数（方向和距离）
    theta = [0]  # 方向，可以是0, 45, 90度等
    distance = [1]  # 距离，可以是1, 2, 3等

    # 计算GLCM
    glcm = graycomatrix(original_image_array, distance, theta, symmetric=True, normed=True)

    # 计算GLCM特定参数下的sum average
    glcm_theta = glcm[distance, theta]
    sum_average = np.sum(glcm_theta * (np.arange(2, 2 * len(glcm_theta) + 2)))

    # 创建增强图像
    enhanced_image = np.zeros_like(original_image_array, dtype=np.float32)
    window_size = 5

    for i in range(original_image_array.shape[0] - window_size + 1):
        for j in range(original_image_array.shape[1] - window_size + 1):
            window = original_image_array[i:i + window_size, j:j + window_size]
            sum_squares = np.sum(window ** 2)
            enhanced_image[i + window_size // 2, j + window_size // 2] = sum_squares

    # 可以根据需要对增强图像进行归一化或其他后处理
    # 归一化增强图像像素值到0到1的范围
    enhanced_image = rescale_intensity(enhanced_image, in_range='image', out_range=(0, 1))

    # 将增强图像数据类型转换为整数并保存为图像文件
    enhanced_image = img_as_ubyte(enhanced_image)

    # 保存增强后的图像
    os.makedirs("../enhanced_image/train/benign", exist_ok=True)
    cv2.imwrite('../enhanced_image/train/benign/' + filename, enhanced_image)
