import os
import shutil

# 定义XML文件夹、图像文件夹和目标文件夹
xml_folder = 'path/to/xml/folder'
image_folder = 'path/to/image/folder'
target_folder = 'path/to/target/folder'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历XML文件夹中的所有文件
for root, dirs, files in os.walk(xml_folder):
    for file in files:
        if file.endswith('.xml'):
            xml_file_path = os.path.join(root, file)
            
            # 获取XML文件的基本名称（不带扩展名）
            xml_base_name = os.path.splitext(file)[0]
            
            # 构建对应的图像文件名
            image_filename = f"{xml_base_name}.bmp"
            image_file_path = os.path.join(image_folder, image_filename)
            
            # 检查图像文件是否存在
            if os.path.exists(image_file_path):
                # 构建目标文件路径
                target_image_path = os.path.join(target_folder, image_filename)
                
                # 复制图像文件到目标文件夹
                shutil.copy(image_file_path, target_image_path)
                print(f"Copied {image_file_path} to {target_image_path}")
            else:
                print(f"Image file {image_file_path} does not exist.")

print("Done.")