import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 定义一个函数来解析XML文件并返回标注框的面积和长宽比
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    areas = []
    aspect_ratios = []

    for object in root.findall(".//object"):
        polygon = object.find("polygon")
    if polygon is not None and polygon.text:
        points = polygon.text.strip().split()
        x = [float(points[i]) for i in range(0, len(points), 2)]
        y = [float(points[i]) for i in range(1, len(points), 2)]

        area = 0.5 * abs(sum(x[i] * (y[i + 1] - y[i - 1]) for i in range(len(x))) + x[-1] * (y[0] - y[-1]))
        width = max(x) - min(x)
        height = max(y) - min(y)

        areas.append(area)
        aspect_ratios.append(width / height)

    return areas, aspect_ratios

# 指定存储XML文件的文件夹路径
xml_folder = "/home/quchenyu/DUONet/datasets/ShipRSImageNet_V1/Annotations"

all_areas = []
all_aspect_ratios = []

# 遍历文件夹下的所有XML文件
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, xml_file)
        areas, aspect_ratios = parse_xml(xml_path)
        all_areas.extend(areas)
        all_aspect_ratios.extend(aspect_ratios)

# 绘制标注框面积的直方图
plt.hist(all_areas, bins=50, range=(0, max(all_areas)))
plt.title("标注框面积分布")
plt.xlabel("面积")
plt.ylabel("数量")
plt.show()

# 绘制标注框长宽比的直方图
plt.hist(all_aspect_ratios, bins=50, range=(0, max(all_aspect_ratios)))
plt.title("标注框长宽比分布")
plt.xlabel("长宽比")
plt.ylabel("数量")
plt.show()
