def compare_and_remove(source_txt_1, source_txt_2, target_txt):
    # 读取两个源txt文件中的内容
    with open(source_txt_1, 'r') as source_file_1:
        source_content_1 = set(source_file_1.read().splitlines())

    with open(source_txt_2, 'r') as source_file_2:
        source_content_2 = set(source_file_2.read().splitlines())

    # 读取目标txt文件中的内容
    with open(target_txt, 'r') as target_file:
        target_content = set(target_file.read().splitlines())

    # 在源txt文件中删除与目标txt相同的名称
    updated_source_1 = source_content_1.difference(target_content)
    updated_source_2 = source_content_2.difference(target_content)

    # 更新源txt文件
    with open(source_txt_1, 'w') as source_file_1:
        source_file_1.write('\n'.join(updated_source_1))

    with open(source_txt_2, 'w') as source_file_2:
        source_file_2.write('\n'.join(updated_source_2))

# 设置两个源txt文件和一个目标txt文件的路径
source_txt_1 = "/home/quchenyu/DUONet/make/DOSR/ImageSets/trainval copy12.txt"
source_txt_2 = "//home/quchenyu/DUONet/make/DOSR/ImageSets/trainval copy12.txt"
target_txt = "/home/quchenyu/DUONet/make/10+10_5/all_files_to_delete.txtt"

# 执行对比和删除操作
compare_and_remove(source_txt_1, source_txt_2, target_txt)
