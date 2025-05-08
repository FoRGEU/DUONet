import os
import shutil

# 定义原始文件夹和目标文件夹
source_folder = "/home/quchenyu/DUONet/make/50+0/Annotations"
target_folder = "/home/quchenyu/DUONet/make/42+8/Annotations"

# 从txt文件中读取文件名
txt_file = "/home/quchenyu/DUONet/make/50+0/ImageSets/Main/val.txt"  # 替换为你的txt文件路径
with open(txt_file, "r") as file:
    file_names = file.readlines()

# 去除每行末尾的换行符并添加.xml后缀
file_names = [name.strip() + ".xml" for name in file_names]

# 复制XML文件到目标文件夹
for xml_file in file_names:
    source_path = os.path.join(source_folder, xml_file)
    target_path = os.path.join(target_folder, xml_file)
    shutil.copy(source_path, target_path)

print("XML文件已复制到目标文件夹。")
