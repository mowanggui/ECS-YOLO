import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打开表
df = pd.read_excel("E:\其他结果\ECS-LIF.xlsx")

# 输入折线图数据
plt.plot(df["noise intensities"], df["LIFmAP_0.5"], label='LIF', linewidth=1, color='c', marker='o', markerfacecolor='blue',
         markersize=5)
# 横坐标为物品编号，纵坐标为库存量，线的名称为库存量，粗细为1，颜色为青色，标记为“o”所代表的图形（会在后面详细介绍），颜色为蓝色，大小为5
plt.plot(df["noise intensities"], df["ECSLIFmAP_0.5"], label='ECS-LIF', linewidth=1, color='y', marker='o', markerfacecolor='blue',
         markersize=5)
# plt.plot(df["epoch"], df["CGFMv21mAP_0.5"], label='CGFMv2noATT', linewidth=1, color='r', marker='v',
#          markerfacecolor='blue', markersize=5)
# plt.plot(df["epoch"], df["CGFMv22mAP_0.5"], label='CGFMv2ATT', linewidth=1, color='m', marker='1',
#          markerfacecolor='blue', markersize=5)
plt.xlabel("noise intensities")
# 横坐标为物品编号
plt.ylabel('mAP@0.5')
# 纵坐标为各类指标
# plt.title("Comparison of ECS-LIF and LIF at different noise intensities")
# 折线图的名称


# 图例说明
plt.legend()
# 显示网格
plt.grid()
plt.savefig("ECS-LIF2.png", dpi=600, bbox_inches='tight', pad_inches=0)
# 显示图像
plt.show()
