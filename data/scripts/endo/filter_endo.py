
### 去除非胃镜的图片，如切除后的肿瘤

import os

raw_data_path = '/home/jc/workspace/MedLLMs/Eso-Llava/data/raw_data/endo/precancer/unlabeled'

filtered_filename = eval(open('/home/jc/workspace/MedLLMs/Eso-Llava/data/raw_data/endo/precancer/unlabeled/ls.txt', 'r').read())
print(">>> Filtered filename type:", type(filtered_filename))
print(">>> Filtered filename len:", len(filtered_filename))

total_file_num = 0
for dirpath, dirnames, filenames in os.walk(raw_data_path):
    # print("Dirpath, dirnames, filenames:", dirpath, dirnames, filenames)
    if len(dirnames) == 0:  # 到达文件夹底部
        for filename in filenames:
            total_file_num += 1
            if filename not in filtered_filename:
                print(">>> TB delete, Processing:", dirpath+'/'+filename)
                os.remove(dirpath+'/'+filename)
print(">>> Total file number:", total_file_num)
