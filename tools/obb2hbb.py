import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def obb2hbb(obb):
    cx, cy, w, h, angle = obb
    theta = np.radians(angle)

    # Calculate the four corners of the OBB
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate and translate the corners
    rotated_corners = np.dot(corners, rotation_matrix) + np.array([cx, cy])

    # Find the min and max points
    x_min, y_min = np.min(rotated_corners, axis=0)
    x_max, y_max = np.max(rotated_corners, axis=0)

    return [x_min, y_min, x_max, y_max]

# 示例用法
obb_params = (0, 0, 4, 2, 30)  # 以(cx, cy, width, height, angle)形式给定的OBB参数
hbb_params = obb2hbb(obb_params)

# 创建图形
fig, ax = plt.subplots()

# 绘制原始的OBB
corners = np.array([
    [-obb_params[2]/2, -obb_params[3]/2],
    [obb_params[2]/2, -obb_params[3]/2],
    [obb_params[2]/2, obb_params[3]/2],
    [-obb_params[2]/2, obb_params[3]/2]
])
rotated_corners = np.dot(corners, np.array([
    [np.cos(np.radians(obb_params[4])), -np.sin(np.radians(obb_params[4]))],
    [np.sin(np.radians(obb_params[4])), np.cos(np.radians(obb_params[4]))]
])) + np.array([obb_params[0], obb_params[1]])
obb_patch = Polygon(rotated_corners, closed=True, edgecolor='b', facecolor='none', label='OBB')
ax.add_patch(obb_patch)

# 绘制转换后的HBB
hbb_patch = Rectangle((hbb_params[0], hbb_params[1]), hbb_params[2] - hbb_params[0], hbb_params[3] - hbb_params[1],
                      edgecolor='r', facecolor='none', label='HBB')
ax.add_patch(hbb_patch)

# 设置图形属性
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('OBB to HBB Conversion')
plt.legend()
plt.grid(True)
plt.show()
