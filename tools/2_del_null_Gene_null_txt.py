import os
import shutil
import xml.etree.ElementTree as ET

def contains_object(xml_file_path):
    # 解析XML文件并检查是否包含<object>元素
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        if root.find("object") is not None:
            return True
    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file_path}")
    return False

def delete_files_without_object_and_generate_txt(source_folder, output_txt_path):
    # 初始化总计数器
    total_deleted_count = 0
    files_to_delete = []

    # 遍历源文件夹中的所有XML文件并删除不含<object>的文件
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)
                if not contains_object(xml_file_path):
                    # 记录将要删除的文件名
                    files_to_delete.append(os.path.splitext(file)[0])
                    # 删除不含<object>的文件
                    os.remove(xml_file_path)
                    total_deleted_count += 1
                    print(f"Deleted {file}")

    # 将要删除的文件名写入txt文件（不带后缀）
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write('\n'.join(files_to_delete))

    return total_deleted_count

# 设置源文件夹和输出txt文件的路径
source_folder_path = "/home/quchenyu/DUONet/make/10+10_5/Annotations"
output_txt_path = "/home/quchenyu/DUONet/make/10+10_5/all_files_to_delete.txt"

# 执行文件删除操作并生成txt文件
total_deleted_count = delete_files_without_object_and_generate_txt(source_folder_path, output_txt_path)

# 打印总共删除的文件数量
print("Total deleted files without <object>: {}".format(total_deleted_count))
