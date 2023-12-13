import os
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution

def multi_objective_function(gamma_values, image):
    gamma1, gamma2 = gamma_values

    # 应用伽马校正
    temp = np.power(image, gamma1)
    corrected_image = np.power(temp, gamma2)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    # 计算伽马校正后图像的直方图
    hist, _ = np.histogram(corrected_image, bins=256, range=[0, 256], density=True)

    # 计算信息熵
    entropy_value = -np.sum(hist * np.log2(hist + 1e-10))

    # 使用 Sobel 边缘检测算子
    sobel_x = cv2.Sobel(corrected_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(corrected_image, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_content = np.sum(edges) / (corrected_image.shape[0] * corrected_image.shape[1])

    # 计算灰度标准方差
    std_dev = np.std(corrected_image)

    # 赋予不同指标不同权重
    weight_entropy = 0.3
    weight_edge_content = 0.3
    weight_std_dev = 0.3

    # 计算综合评估值
    objective_value = (
        weight_entropy * entropy_value +
        weight_edge_content * edge_content +
        weight_std_dev * std_dev
    )

    return -objective_value  # 取负号，因为通常是最小化目标

def apply_gamma(image, gamma1, gamma2):
    # 第一次伽马校正
    temp = np.power(image, gamma1)
    # 第二次伽马校正
    output = np.power(temp, gamma2)

    # 将图像的像素值归一化到 0 到 255 的范围
    output = ((output - np.min(output)) / (np.max(output) - np.min(output)) * 255).astype(np.uint8)

    # 将NumPy数组转换为Image对象
    enhanced_image = Image.fromarray(output)

    return enhanced_image


def optimize_gamma(image):
    # 定义伽马值的搜索范围
    bounds = [(0, 3), (0, 3)]

    # 使用差分进化算法进行优化
    result = differential_evolution(multi_objective_function, bounds, args=(image,), strategy='best1bin', popsize=20, tol=0.01,
                                    mutation=(0.8, 0.8), recombination=0.8, seed=42)

    # 获取最优的伽马值
    optimal_gamma_values = result.x

    return optimal_gamma_values


def apply_gamma_to_folder(input_folder, output_folder):
    # 创建保存目录
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入目录中的每个子目录（benign和malignant）
    for class_folder in os.listdir(input_folder):
        class_input_path = os.path.join(input_folder, class_folder)
        class_output_path = os.path.join(output_folder, class_folder)

        # 创建保存增强图像的子目录
        os.makedirs(class_output_path, exist_ok=True)

        # 遍历每个子目录中的图像文件
        for file_name in os.listdir(class_input_path):
            file_path = os.path.join(class_input_path, file_name)
            output_file_path = os.path.join(class_output_path, file_name)

            # 读取图像
            image = Image.open(file_path)

            # 优化伽马值
            optimal_gamma_values = optimize_gamma(np.array(image))

            # 应用优化后的伽马值
            enhanced_image = apply_gamma(np.array(image), *optimal_gamma_values)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "gamma_" + file_name))


if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/train"
    output_folder = "../data/一期gamma/train"

    # 应用gamma增强
    apply_gamma_to_folder(input_folder, output_folder)
