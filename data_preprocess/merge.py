import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
     # 读取Excel文件
     excel_file_path = '../data/val.xlsx'
     df = pd.read_excel(excel_file_path)

     # 创建新列，表示预测是否正确
     df['correct_before'] = df['prob'] == df['label']
     df['correct_after'] = df['prob_clahe'] == df['label']

     # 创建新列，表示预测结果的变化
     df['change'] = df['correct_before'] != df['correct_after']

     # 进一步细分为前边预测正确、后边预测错误和前错后对两种情况
     df['correct_before_and_after'] = (df['correct_before'] & df['correct_after']).astype(int)
     df['correct_before_wrong_after'] = (df['correct_before'] & ~df['correct_after']).astype(int)
     df['wrong_before_correct_after'] = (~df['correct_before'] & df['correct_after']).astype(int)
     df['wrong_before_and_after'] = (~df['correct_before'] & ~df['correct_after']).astype(int)

     # 统计各类情况的样本数量
     correct_before_count = df['correct_before'].sum()
     correct_after_count = df['correct_after'].sum()
     change_count = df['change'].sum()
     correct_before_and_after_count = df['correct_before_and_after'].sum()
     correct_before_wrong_after_count = df['correct_before_wrong_after'].sum()
     wrong_before_correct_after_count = df['wrong_before_correct_after'].sum()
     wrong_before_and_after_count = df['wrong_before_and_after'].sum()

     # 输出统计结果
     print(f"增强前预测正确的样本数量：{correct_before_count}")
     print(f"增强后预测正确的样本数量：{correct_after_count}")
     print(f"预测结果变化的样本数量：{change_count}")
     print(f"前边预测正确、后边预测正确的样本数量：{correct_before_and_after_count}")
     print(f"前边预测正确、后边预测错误的样本数量：{correct_before_wrong_after_count}")
     print(f"前边预测错误、后边预测正确的样本数量：{wrong_before_correct_after_count}")
     print(f"前后都预测错误的样本数量：{wrong_before_and_after_count}")

     # 设置中文显示
     plt.rcParams['font.sans-serif'] = ['SimHei']
     plt.rcParams['axes.unicode_minus'] = False

     # 可视化
     plt.figure(figsize=(12, 8))

     ax = sns.barplot(
          x=['前边预测正确', '后边预测正确', '前后都正确','前后都预测错误','预测结果变化', '前对后错','前错后对'],
          y=[correct_before_count, correct_after_count,
             correct_before_and_after_count, wrong_before_and_after_count,
             change_count,correct_before_wrong_after_count,
             wrong_before_correct_after_count])

     # 在每个柱形上标注数量
     for p in ax.patches:
          ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')

     plt.title('预测结果变化情况详细分析')
     plt.xlabel('情况')
     plt.ylabel('样本数量')
     plt.show()

     df.to_excel('../data/new_val.xlsx', index=False)