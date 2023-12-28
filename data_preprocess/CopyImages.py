import os
import shutil

def copy_images(src_folder, dest_folder, corresponding_folder):
    for root, dirs, files in os.walk(corresponding_folder):
        for file in files:
            src_path = os.path.join(src_folder, os.path.relpath(root, corresponding_folder), file)  # 构造波段二图片路径
            dest_path = os.path.join(dest_folder, os.path.relpath(root, corresponding_folder), file)
            corresponding_path = os.path.join(corresponding_folder, os.path.relpath(root, corresponding_folder), file)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # 如果目标路径已存在相同的文件，则进行复制
            if os.path.exists(src_path) and os.path.exists(corresponding_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied {file} to {dest_path}")

if __name__ == "__main__":
    src_folder = "../data/ti_二期双十+双十五"
    dest_folder = "../data/ti_二期双十+双十五wave1"
    corresponding_folder = "../data/ti_二期双十+双十五wave2"  # 假设这是波段一的文件夹

    copy_images(src_folder, dest_folder, corresponding_folder)
