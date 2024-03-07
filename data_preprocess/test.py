import os
from PIL import Image
from openpyxl import Workbook

def get_bmp_filenames(folder_path):
    bmp_filenames = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                # 使用PIL库检查文件是否是BMP格式
                with Image.open(filepath) as img:
                    if img.format == 'BMP':
                        bmp_filenames.append(filename)
            except Exception as e:
                # 如果文件不是图像或者无法打开，忽略错误
                pass
    return bmp_filenames

def save_to_excel(bmp_filenames, excel_filename='bmp_filenames.xlsx'):
    wb = Workbook()
    ws = wb.active
    ws.append(['BMP Filenames'])

    for bmp_filename in bmp_filenames:
        ws.append([bmp_filename])

    wb.save(excel_filename)

if __name__ == "__main__":
    folder_path = '../data/ti_二期双十wave1/val/benign'  # 将'/path/to/your/folder'替换为你的文件夹路径
    bmp_filenames = get_bmp_filenames(folder_path)

    if bmp_filenames:
        save_to_excel(bmp_filenames)
        print(f'BMP filenames saved to excel file: bmp_filenames.xlsx')
    else:
        print('No BMP files found in the specified folder.')
