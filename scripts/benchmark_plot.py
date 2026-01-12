import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['PCL-Reasoner-V1.5', 'PCL-Reasoner-V1', 'Qwen3-32B', 'QwQ-32B', 'Skywork-OR1-32B', 'AM-Thinking-v1']
aime_2024 = [92.0, 85.7, 81.4, 79.5, 82.2, 85.3]
aime_2025 = [90.6, 84.2, 72.9, 69.5, 73.3, 74.4]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 定义 x 轴位置（两个组：2024 和 2025）
x = np.array([0, 1])  # 两个年份
bar_width = 0.12

# 颜色和图案设置
colors = ['#ffcc00', '#8bc34a', '#b8d8c3', '#9bc9a9', '#2e5f6c', '#2b3d4b']  # 新颜色放在首位
hatches = ['/', '/', '', '', '', '']  # 第一个和第二个模型都使用斜线图案

# 绘制柱子
for i in range(len(models)):
    pos1 = x[0] + (i - 2.5) * bar_width
    pos2 = x[1] + (i - 2.5) * bar_width
    ax.bar(pos1, aime_2024[i], bar_width, color=colors[i], hatch=hatches[i], linewidth=0.5)
    ax.bar(pos2, aime_2025[i], bar_width, color=colors[i], hatch=hatches[i], linewidth=0.5)

# 添加数值标签
for i in range(len(models)):
    pos1 = x[0] + (i - 2.5) * bar_width
    pos2 = x[1] + (i - 2.5) * bar_width
    if i == 0:
        ax.text(pos1, aime_2024[i] + 0.5, f'{aime_2024[i]}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(pos2, aime_2025[i] + 0.5, f'{aime_2025[i]}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    else:
        ax.text(pos1, aime_2024[i] + 0.5, f'{aime_2024[i]}', ha='center', va='bottom', fontsize=12)
        ax.text(pos2, aime_2025[i] + 0.5, f'{aime_2025[i]}', ha='center', va='bottom', fontsize=12)

# 设置坐标轴
# ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=15)
ax.set_ylim(65, 102)
ax.set_yticks(range(70, 102, 10))
# ax.set_yticklabels([f'{i}%' for i in range(45, 101, 10)], fontweight='bold')
ax.set_yticklabels([f'{i}%' for i in range(70, 102, 10)], fontsize=15)
ax.set_xticks(x)
# ax.set_xticklabels(['AIME 2024', 'AIME 2025'], fontsize=15, fontweight='bold')
ax.set_xticklabels(['AIME 2024', 'AIME 2025'], fontsize=15)

# 使用 fig.legend() 实现全宽图例
legend_handles = []
for i, model in enumerate(models):
    if i == 0:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], hatch='/', label=model))
    else:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=model))

# 图例横跨整个图宽，位于顶部
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False,
           columnspacing=2.0, handletextpad=0.5, handlelength=3, fontsize=18)

# 去掉标题（已无 title）
plt.title('')

# 调整布局：增加顶部边距以容纳图例
plt.tight_layout(rect=[0, 0, 1, 0.80])  # 保留顶部 5% 空间给图例

# 保存和显示
plt.savefig('benchmark.png', dpi=300, bbox_inches='tight')
plt.show()
plt.savefig('images/benchmark.png')
