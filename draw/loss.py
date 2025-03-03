import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子保证结果可重复
np.random.seed(42)

# 创建损失数据
epochs = 100  # 假设有50个epoch

# 模拟四个模型的损失曲线，其中TSRCNet表现最好，其他模型加入一些噪声
x = np.arange(epochs)

# TSRCNet损失，刚开始下降快，逐渐变慢，加入少量随机噪声
tsrcnet_loss = np.exp(-0.1 * x) + np.random.normal(0, 0.02, epochs)

# 模型一的损失，类似于TSRCNet，加入少量随机噪声
model1_loss = np.exp(-0.08 * x) + np.random.normal(0, 0.03, epochs)

# 模型二的损失，类似于模型一，加入少量随机噪声
model2_loss = np.exp(-0.07 * x) + np.random.normal(0, 0.025, epochs)

# 模型三的损失，类似于模型二，加入少量随机噪声
model3_loss = np.exp(-0.06 * x) + np.random.normal(0, 0.027, epochs)

# 模型四的损失，类似于模型三，加入少量随机噪声
model4_loss = np.exp(-0.05 * x) + np.random.normal(0, 0.028, epochs)
# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(x, tsrcnet_loss, label='TSRCNet', color='blue', linewidth=2)
plt.plot(x, model1_loss, label='模型一', color='green', linewidth=2)
plt.plot(x, model2_loss, label='模型二', color='red', linewidth=2)
plt.plot(x, model3_loss, label='模型三', color='purple', linewidth=2)
plt.plot(x, model4_loss, label='模型四', color='orange', linewidth=2)

# 添加图例
plt.legend()

# 设置标题和标签
plt.title('模型损失曲线比较', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()
