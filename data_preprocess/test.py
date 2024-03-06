import os
from PIL import Image

def find_different_images(folder1, folder2):
    # 获取两个文件夹中的文件列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找到文件名不相同的图片
    different_images = files1.symmetric_difference(files2)

    # 输出不同的图片文件名
    if different_images:
        print("文件名不相同的图片:")
        for image_name in different_images:
            print(f"- {image_name}")
    else:
        print("两个文件夹中的图片文件名完全相同。")

def main():
    # 替换为你的文件夹路径
    folder1_path = "../data/省肿瘤493wave1"
    folder2_path = "../data/省肿瘤493wave1原始图"

    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        print("指定的文件夹路径不存在。请检查路径并重新运行程序。")
        return

    find_different_images(folder1_path, folder2_path)

if __name__ == "__main__":
    main()
