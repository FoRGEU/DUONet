import os
import shutil



import os
import shutil

# 定义txt文件路径和图片文件夹路径
# 定义txt文件路径和图片文件夹路径
txt_file_path = '/home/quchenyu/DUONet/make/10+10_5/ImageSets/Main/test.txt'
image_folder_path = '/home/quchenyu/DUONet/make/DOSR/JPEGImages'

# 定义目标文件夹路径，用于存放提取出的图片
output_folder_path = '/home/quchenyu/DUONet/datasets/test_png_DOSR'

# 创建目标文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 读取txt文件内容
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# 提取对应的图片
for line in lines:
    # 去除换行符
    image_name = line.strip() + '.png'  # 添加后缀为.png
    
    # 构建图片的完整路径
    image_path = os.path.join(image_folder_path, image_name)
    
    # 检查图片是否存在
    if os.path.exists(image_path):
        # 构建目标文件路径
        output_path = os.path.join(output_folder_path, image_name)
        
        # 复制图片到目标文件夹
        shutil.copy(image_path, output_path)
        print(f"Image '{image_name}' copied to '{output_path}'")
    else:
        print(f"Image '{image_name}' not found in '{image_folder_path}'")

