from PIL import Image
import os
import random

def rotate_image(image, angle):
    return image.rotate(angle)

def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def scale_image(image, scale_factor):
    new_size = tuple(int(dim * scale_factor) for dim in image.size)
    return image.resize(new_size, Image.ANTIALIAS)

def apply_augmentation_to_folder(input_folder, output_folder):
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

            # 旋转
            angle = 30
            rotated_image = rotate_image(image, angle)

            # 水平翻转
            flipped_horizontal_image = horizontal_flip(image)

            # 垂直翻转
            flipped_vertical_image = vertical_flip(image)

            # 缩放
            scale_factor = random.uniform(1.1, 1.2)
            scaled_image = scale_image(image, scale_factor)

            # 保存增强后的图像
            rotated_image.save(os.path.join(class_output_path, "rotated_" + file_name))
            flipped_horizontal_image.save(os.path.join(class_output_path, "flipped_horizontal_" + file_name))
            flipped_vertical_image.save(os.path.join(class_output_path, "flipped_vertical_" + file_name))
            scaled_image.save(os.path.join(class_output_path, "scaled_" + file_name))

if __name__ == '__main__':
    # 指定原始图像文件夹和保存增强图像的文件夹
    input_folder = "../data/二期双十/train"
    output_folder = "../data/二期双十/new_train"

    # 应用数据增强
    apply_augmentation_to_folder(input_folder, output_folder)
