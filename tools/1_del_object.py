import os
import xml.etree.ElementTree as ET

def process_xml_file(input_file, output_folder, names_to_remove):
    # 解析XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()

    # 初始化总计数器
    total_removed_count = 0

    # 查找<object>下的<name>元素，并删除指定的name值
    for name_to_remove in names_to_remove:
        for obj in root.findall(".//object[name='{}']".format(name_to_remove)):
            # 记录删除的次数
            total_removed_count += 1
            root.remove(obj)

    # 构造输出文件路径
    output_file = os.path.join(output_folder, os.path.basename(input_file))

    # 保存修改后的XML文件到新的文件夹下
    tree.write(output_file)

    return total_removed_count

def process_folder(input_folder, output_folder, names_to_remove):
    # 初始化总计数器
    total_removed_count = 0

    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".xml"):
                input_file = os.path.join(root, file)
                total_removed_count += process_xml_file(input_file, output_folder, names_to_remove)

    return total_removed_count

# 指定要处理的输入文件夹路径和输出文件夹路径
input_folder = "/home/quchenyu/DUONet/make/DOSR/trainval_5_annos"
output_folder = "/home/quchenyu/DUONet/make/10+10_5/Annotations"

# 要删除的name值列表
names_to_remove = ["military_ship", 'container', 'deckship', 'barge', 'tanker', 'cargo',
    'cruise', 'submarine', 'tug', 'multihull',  ]

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 处理输入文件夹下的所有XML文件，并保存到输出文件夹下
total_removed_count = process_folder(input_folder, output_folder, names_to_remove)

# 打印总共删除的标注信息数量
print("Total deleted annotations: {}".format(total_removed_count))


