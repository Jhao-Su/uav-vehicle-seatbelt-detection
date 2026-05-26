#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
使用说明
-------
本脚本用于从图片和标注文件中筛选出有效的"图片-标注"对。
具体规则：图片文件必须存在对应的标注文件（.txt），且标注文件内容不能为空。

使用方法：
    1. 修改 IMAGES_SRC_DIR 变量为原始图片所在目录路径
    2. 修改 LABELS_SRC_DIR 变量为原始标注文件所在目录路径
    3. 修改 IMAGES_DST_DIR 变量为筛选后图片的输出目录路径
    4. 修改 LABELS_DST_DIR 变量为筛选后标注文件的输出目录路径
    5. 运行脚本：
       python clean_empty_data.py

输出：
    筛选后，有效的图片和标注文件分别保存到指定的输出目录中。
    同时会在控制台打印统计信息（原始图片数、有效对数、缺失标注数、空标注数等）。

依赖：
    标准库 pathlib、shutil，无需额外安装。
'''

from pathlib import Path
import shutil

# 替换为实际路径
IMAGES_SRC_DIR = Path("path/to/valid/images")
LABELS_SRC_DIR = Path("path/to/valid/labels_converted")
IMAGES_DST_DIR = Path("path/to/valid/images_fixed")
LABELS_DST_DIR = Path("path/to/valid/labels_fixed")

print("=" * 70)
print("📁 筛选有效图片 - 标注对")
print("=" * 70)

if IMAGES_DST_DIR.exists():
    shutil.rmtree(IMAGES_DST_DIR)
if LABELS_DST_DIR.exists():
    shutil.rmtree(LABELS_DST_DIR)

IMAGES_DST_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DST_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(IMAGES_SRC_DIR.glob("*.jpg"))
valid_pairs = 0
missing_labels = 0
empty_labels = 0

for img_file in image_files:
    label_file = LABELS_SRC_DIR / (img_file.stem + ".txt")
    
    if not label_file.exists():
        missing_labels += 1
        continue
    
    with open(label_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        empty_labels += 1
        continue
    
    shutil.copy2(img_file, IMAGES_DST_DIR / img_file.name)
    shutil.copy2(label_file, LABELS_DST_DIR / label_file.name)
    valid_pairs += 1

print(f"原始图片数：{len(image_files)}")
print(f"有效图片 - 标注对：{valid_pairs}")
print(f"缺失标注文件：{missing_labels}")
print(f"标注文件为空：{empty_labels}")
print(f"\nimages_fixed: {len(list(IMAGES_DST_DIR.glob('*.jpg')))}")
print(f"labels_fixed: {len(list(LABELS_DST_DIR.glob('*.txt')))}")
print("\n✅ 完成！")
print("=" * 70)