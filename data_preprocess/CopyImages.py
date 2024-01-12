import os
import shutil

# 定义路径
source_folder = "../data/第一波段原始图"
target_folder = "../data/ti_二期双十+双十五wave1原始图"

# 获取原始图文件名列表
source_images = os.listdir(source_folder)

# 遍历 ti_二期双十+双十五wave1 中的 train、val 和 test 文件夹
for set_folder in ["train", "val", "test"]:
    set_path = os.path.join("../data/ti_二期双十+双十五wave1", set_folder)

    # 遍历 benign 和 malignant 文件夹
    for class_folder in ["benign", "malignant"]:
        class_path = os.path.join(set_path, class_folder)

        # 遍历每个文件夹中的图像文件
        for image_name in os.listdir(class_path):
            # 检查第二波段原始图中是否存在相同文件名的图像
            if image_name in source_images:
                # 构建源文件和目标文件路径
                source_path = os.path.join(source_folder, image_name)
                target_path = os.path.join(target_folder, set_folder, class_folder, image_name)
                new_target_folder = os.path.join(target_folder, set_folder, class_folder)
                # 确保保存文件夹存在，如果不存在则创建
                if not os.path.exists(source_folder):
                    os.makedirs(source_folder)

                if not os.path.exists(new_target_folder):
                    os.makedirs(new_target_folder)
                # 复制文件
                shutil.copy(source_path, target_path)

print("复制完成。")