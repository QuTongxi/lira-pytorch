import matplotlib.pyplot as plt
import numpy as np

# 数据

scale = 10
labels = ['AUC ', 'AUC (q)', 'Accuracy ', 'Accuracy (q)', 'TPR(10x)', 'TPR(10x)(q)']
values = [0.5798, 0.5643, 0.5788, 0.5598, 0.0257 * scale, 0.0369 * scale]

# 设置条形位置
x = np.arange(len(labels))

# 创建条形图
plt.bar(x, values, color=['blue', 'orange']*3, width=0.4)

# 添加标签和标题
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('attack ours (offline, fixed variance)')
plt.xticks(x, labels, rotation=30, ha='right')
plt.ylim(0, 0.7)  # 设置y轴的范围
# 在柱上显示数值
for i in range(len(values)):
    if i <= 3:
        plt.text(x[i], values[i] + 0.01, f'{values[i]:.4f}', ha='center', va='bottom')
    else:
       plt.text(x[i], values[i] + 0.01, f'{values[i]/scale:.4f}', ha='center', va='bottom') 
# 显示图形
plt.tight_layout()
plt.show()
plt.savefig("./qtx.png")
