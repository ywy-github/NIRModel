import os
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix
from skimage import img_as_ubyte

def apply_GLCM(image):
    # 将图像复制到一个NumPy数组，并将其数据类型转换为可写的
    original_image_array = np.array(image, dtype=np.uint8)

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
    enhanced_image = rescale_intensity(enhanced_image, in_range='image', out_range=(-1, 1))

    # 将增强图像数据类型转换为整数
    enhanced_image = img_as_ubyte(enhanced_image)

    # 将NumPy数组转换为Image对象
    enhanced_image_pil = Image.fromarray(enhanced_image)

    return enhanced_image_pil

def apply_glcm_to_folder(input_folder, output_folder):
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

            # 应用GLCM增强
            enhanced_image = apply_GLCM(image)

            # 保存增强后的图像
            enhanced_image.save(os.path.join(class_output_path, "glcm_" + file_name))


if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/一期数据/test"
    output_folder = "../data/一期GLCM/test"

    # 应用GLCM增强
    apply_glcm_to_folder(input_folder, output_folder)
