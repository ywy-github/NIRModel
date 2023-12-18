import os
import cv2
import numpy as np
from PIL import Image


def calculate_area(image):
    gray_image = np.array(image)
    # 计算二值图像中白色（有效区域）的像素数量
    return cv2.countNonZero(gray_image)


def compare_and_remove(original_folder, otsu_folder):
    # 遍历原始文件夹中的图像
    for root, dirs, files in os.walk(original_folder):
        for file_name in files:
            if file_name.endswith(".bmp"):  # 根据实际文件格式进行调整
                original_path = os.path.join(root, file_name)

                # 获取原图有效区域
                original_image = Image.open(original_path)
                original_area = calculate_area(original_image)

                # 获取使用大津法后的有效光照区域图
                # 获取使用大津法后的有效光照区域图
                otsu_name = "otsu_" + file_name
                relative_path = os.path.relpath(original_path, original_folder)
                otsu_path = os.path.join(otsu_folder, relative_path.replace(os.path.sep,
                                                                            os.path.sep + "otsu_"))  # Fix path construction
                otsu_image = Image.open(otsu_path)

                if otsu_image is not None:
                    # 获取大津法后的有效光照区域
                    otsu_area = calculate_area(otsu_image)

                    # 比较有效光照区域大小
                    if otsu_area < 0.5 * original_area:
                        # 删除原图
                        os.remove(original_path)
                        print(f"Removed {file_name} due to insufficient effective lighting area.")
                else:
                    print(f"Error processing {file_name}")


if __name__ == "__main__":
    original_folder = "../data/一期数据/ti_test"  # 原始图像文件夹
    otsu_folder = "../data/一期数据/otsu_test"  # 使用大津法获取有效光照区域后的图像文件夹

    compare_and_remove(original_folder, otsu_folder)