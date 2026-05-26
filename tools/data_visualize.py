'''
YOLO 格式数据可视化脚本

功能：
  在图片上绘制 YOLO 格式的标注框和类别名称，用于数据可视化检查。

使用方法：
  1. 基本用法（自动寻找 data.yaml 配置文件）：
     python tools/data_visualize.py <图片路径>

     示例：
     python tools/data_visualize.py ../dataset/coco/train/images/image1.jpg

  2. 手动指定 data.yaml 配置文件：
     python tools/data_visualize.py <图片路径> --yaml <yaml路径>

     示例：
     python tools/data_visualize.py ../dataset/coco/train/images/image1.jpg --yaml ../dataset/coco/coco2017_data.yaml

  3. 指定输出路径：
     python tools/data_visualize.py <图片路径> --output <输出路径>

     示例：
     python tools/data_visualize.py ../dataset/coco/train/images/image1.jpg --output result.jpg --yaml ../dataset/coco/coco2017_data.yaml

  4. 手动指定标注文件路径：
     python tools/data_visualize.py <图片路径> --label <标注文件路径>

     示例：
     python tools/data_visualize.py ../dataset/coco/train/images/image1.jpg --label ../dataset/coco/train/labels/image1.txt --yaml ../dataset/coco/coco2017_data.yaml

参数说明：
  image_path    必填，要可视化的图片路径
  --yaml        可选，data.yaml 配置文件路径（不指定则自动向上查找）
  --label       可选，YOLO 标注文件 (.txt) 路径（不指定则自动寻找同名 txt 文件）
  --output      可选，输出图片路径（不指定则默认生成 xxx_vis.jpg）

注意事项：
  - 图片文件需为 YOLO 格式数据集的一部分
  - 标注文件 (.txt) 需与图片同名且在同一目录
  - data.yaml 文件需包含 nc（类别数）和 names（类别名称）字段
  - 项目中提供的数据集文件夹中的yaml文件做了针对性命名，如果使用本程序对其进行可视化，需要指定对应的yaml文件路径，或修改其下的yaml文件名，可视化完成后请恢复原始文件名
'''

import os
import sys
import cv2
import yaml
import argparse


def load_yaml(yaml_path):
    """加载 data.yaml 文件，返回类别数和类别名称列表。"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    nc = data.get('nc', None)
    names = data.get('names', None)
    if names is None:
        raise ValueError(f"未在 {yaml_path} 中找到 'names' 字段")
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    return nc, names


def find_data_yaml(image_path):
    """根据图片路径自动寻找对应的 data.yaml 文件。"""
    image_path = os.path.abspath(image_path)
    current_dir = os.path.dirname(image_path)

    while current_dir != os.path.dirname(current_dir):
        yaml_path = os.path.join(current_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            return yaml_path

        for f in os.listdir(current_dir):
            if f.endswith('_data.yaml'):
                return os.path.join(current_dir, f)

        current_dir = os.path.dirname(current_dir)

    raise FileNotFoundError(
        f"无法在 {image_path} 的父目录中找到 data.yaml 文件"
    )


def parse_yolo_label(label_path):
    """解析 YOLO 格式的标注文件，返回标注列表。"""
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def draw_labels(image_path, labels, names, output_path=None):
    """在图片上绘制 YOLO 标注框。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 64, 64), (192, 192, 192), (255, 128, 0), (128, 255, 0),
    ]

    for class_id, x_center, y_center, width, height in labels:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = colors[class_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        label_text = f"{class_name}"
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness)

    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_vis.jpg"

    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存至: {output_path}")
    return output_path


def visualize_single_image(image_path, yaml_path=None, label_path=None, output_path=None):
    """可视化单张图片的 YOLO 标注。"""
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if yaml_path is None:
        yaml_path = find_data_yaml(image_path)
        print(f"自动找到配置文件: {yaml_path}")
    else:
        yaml_path = os.path.abspath(yaml_path)

    _, names = load_yaml(yaml_path)
    print(f"类别数: {len(names)}")
    print(f"类别: {names}")

    if label_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        label_path = os.path.join(image_dir, f"{base_name}.txt")
    else:
        label_path = os.path.abspath(label_path)

    labels = parse_yolo_label(label_path)

    if not labels:
        print(f"警告: 未找到标注文件 {label_path} 或标注为空")

    print(f"找到 {len(labels)} 个标注框")

    output_path = draw_labels(image_path, labels, names, output_path)


def main():
    parser = argparse.ArgumentParser(description="YOLO 格式数据可视化脚本")
    parser.add_argument('image_path', type=str, help='图片路径')
    parser.add_argument('--yaml', type=str, default=None, help='data.yaml 路径（可选，自动寻找）')
    parser.add_argument('--label', type=str, default=None, help='YOLO 标注文件路径（可选，自动寻找同名 txt）')
    parser.add_argument('--output', type=str, default=None, help='输出路径（可选）')

    args = parser.parse_args()

    try:
        visualize_single_image(args.image_path, args.yaml, args.label, args.output)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
