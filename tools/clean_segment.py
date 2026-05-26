'''
使用说明
-------
本脚本用于清洗标注文件，移除 YOLO 标注中包含的多余分割数据。

背景：
    当标注文件混合了检测框数据和分割点数据时，每行可能超过 5 个数值。
    标准 YOLO 检测格式为：class x_center y_center width height（共 5 个数值）。
    如果某行超过 5 个数值，说明后面附带了分割点坐标，脚本会将其截断，仅保留前 5 个检测框数据。

使用方法：
    1. 修改 train_label_dir 变量为训练集标注文件所在目录路径
    2. 修改 val_label_dir 变量为验证集标注文件所在目录路径
    3. 运行脚本：
       python clean_segment.py

输出：
    自动处理指定目录下的所有 .txt 标注文件，将包含多余分割数据的行截断为检测格式。
    控制台会打印处理的文件数和被修正的文件数。

依赖：
    标准库 os、glob，无需额外安装。
'''

import os
import glob

def clean_labels(label_dir):
    '''
    清洗标注文件，移除包含多余分割数据的行。
    '''
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    cleaned_count = 0
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        
        for line in lines:
            parts = list(map(float, line.strip().split()))
            # 标准检测格式应该是 5 个数: class x y w h
            # 如果超过 5 个，说明后面跟着分割点，需要截断
            if len(parts) > 5:
                # 保留前5个
                new_line = " ".join(map(str, parts[:5]))
                new_lines.append(new_line + "\n")
                modified = True
            else:
                new_lines.append(line)
        
        if modified:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            cleaned_count += 1
            
    print(f"清洗完成：处理了 {len(txt_files)} 个文件，修正了 {cleaned_count} 个包含多余分割数据的文件。")

# 分别清洗训练集和验证集
# 替换为实际路径
train_label_dir = "path/to/train/labels"
val_label_dir = "path/to/valid/labels"

if os.path.exists(train_label_dir):
    clean_labels(train_label_dir)
if os.path.exists(val_label_dir):
    clean_labels(val_label_dir)