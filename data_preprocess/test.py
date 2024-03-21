import os

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from openpyxl import Workbook


if __name__ == "__main__":

    # 读取Excel文件
    df = pd.read_excel('../data/qc后样本判断情况.xlsx',sheet_name="test")

    # 过滤年龄在60-70的样本
    filtered_df = df[(df['age'] >= 40) & (df['age'] < 50)]

    # 按照罩杯分组计算预测正确和预测错误的样本数量
    result = filtered_df.groupby(['cup_size', 'prob']).size().unstack(fill_value=0).reset_index()

    # 绘制直方图
    fig, ax = plt.subplots(figsize=(10, 6))

    cup_sizes = result['cup_size'].unique()
    num_cup_sizes = len(cup_sizes)
    bar_width = 0.35
    index = range(num_cup_sizes)

    for i, cup_size in enumerate(cup_sizes):
        correct_count = result[result['cup_size'] == cup_size][0].values[0]
        incorrect_count = result[result['cup_size'] == cup_size][1].values[0]
        ax.bar(i, correct_count, color='green', width=bar_width, label=f'{cup_size} Correct')
        ax.bar(i + bar_width, incorrect_count, color='red', width=bar_width, label=f'{cup_size} Incorrect')

    ax.set_xlabel('Cup Size')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution by Cup Size (Age 40-50)')
    ax.set_xticks([i + bar_width / 2 for i in range(num_cup_sizes)])
    ax.set_xticklabels(cup_sizes)
    ax.legend()
    plt.show()



