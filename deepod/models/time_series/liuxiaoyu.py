import numpy as np
import matplotlib.pyplot as plt

# 设置二次函数的参数
quadratic_params = [
    {"a": -0.011, "b": 1.294, "c": 99.698, "color": "blue", "label": "EAST"},
    {"a": -0.017, "b": 2.452, "c": 10.727, "color": "red", "label": "MID"},
    {"a": -0.013, "b": 1.611, "c": 27.980, "color": "green", "label": "WEST"},
    # {"a": 0,"b": 0.926, "c": 93.825, "color": "yellow", "label": "EAST_NORTH"}
]

# 创建x的值，通常是区间[-10, 10]
x = np.linspace(0, 120, 400)



# 设置x的取值范围
x_min, x_max = 18.955, 91.705

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制每条二次函数并标出极值点
for params in quadratic_params:
    y = params["a"] * x ** 2 + params["b"] * x + params["c"]
    vertex_x = -params["b"] / (2 * params["a"])
    vertex_y = params["a"] * vertex_x ** 2 + params["b"] * vertex_x + params["c"]

    ax.plot(x, y, params["color"], label=params["label"])
    ax.plot(vertex_x, vertex_y, 'o', color=params["color"])
    ax.text(vertex_x, vertex_y, '', color=params["color"])

# 绘制灰色虚线表示x的取值范围
ax.axvline(x=x_min, color='grey', linestyle='--', label='X min')
ax.axvline(x=x_max, color='grey', linestyle='--', label='X max')

# 定义一次函数的参数
m = 0.926  # 斜率
b = 93.825  # 截距
y_linear = m * x + b

# 绘制一次函数的图像
ax.plot(x, y_linear, 'm', label='NORTH_EAST')

# 设置图例
ax.legend(fontsize='small', loc='upper right')

# 设置图表标题和坐标轴标签
ax.set_title('Quadratic Functions with Vertex and Linear Function')
ax.set_xlabel('Intensity')
ax.set_ylabel('ConPM10')

# 显示图表
plt.show()

