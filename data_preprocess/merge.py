import os

from openpyxl.workbook import Workbook

if __name__ == '__main__':

    # 创建一个 Excel 工作簿
    workbook = Workbook()
    sheet = workbook.active

    # 设置标题行
    sheet['A1'] = 'dcm_name'

    # 定义要遍历的文件夹路径
    folder_path = '../data/NIR_Wave2'  # 请将路径替换为实际文件夹的路径

    # 递归遍历文件夹中的文件并将文件名写入 Excel
    def write_file_names_to_excel(path, current_row):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                sheet.cell(row=current_row, column=1, value=item)
                current_row += 1
            elif os.path.isdir(item_path):
                current_row = write_file_names_to_excel(item_path, current_row)
        return current_row

    current_row = 2  # 从第二行开始写入文件名
    current_row = write_file_names_to_excel(folder_path, current_row)

    # 保存 Excel 文件为 .xlsx 格式
    workbook.save('../data/wave2.xlsx')

    print('文件名已保存到 wave2.xlsx')