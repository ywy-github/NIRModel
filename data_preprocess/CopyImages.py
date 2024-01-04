import os

source_folder_wave1 = "../data/ti_二期双十+双十五wave1"
source_folder_original = "../data/ti_二期双十+双十五原始图"

missing_images = []

# 遍历 ti_二期双十+双十五wave1 中的 train、val 和 test 文件夹
for set_folder in ["train", "val", "test"]:
    set_path_wave1 = os.path.join(source_folder_wave1, set_folder)
    set_path_original = os.path.join(source_folder_original, set_folder)

    # 遍历 benign 和 malignant 文件夹
    for class_folder in ["benign", "malignant"]:
        class_path_wave1 = os.path.join(set_path_wave1, class_folder)
        class_path_original = os.path.join(set_path_original, class_folder)

        # 遍历每个文件夹中的图像文件
        for image_name in os.listdir(class_path_wave1):
            # 构建源文件和目标文件路径
            source_path_wave1 = os.path.join(class_path_wave1, image_name)
            source_path_original = os.path.join(class_path_original, image_name)

            # 检查文件是否存在于 ti_二期双十+双十五原始图 中
            if not os.path.exists(source_path_original):
                missing_images.append(image_name)

# 打印缺失的图像文件名
print("以下图像在 ti_二期双十+双十五原始图 中不存在:")
for image_name in missing_images:
    print(image_name)
