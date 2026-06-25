"""
RTDETR 安全带检测模型训练入口 v1

原 train_v1.py 的路径问题修复版：使用 __file__ 解析相对路径，不依赖 CWD。
Docker 环境下同样可用（只需挂载 dataset 目录并传参覆盖默认路径）。

Usage:
    # 本地
    python rtdetr_seatbelt_detection_model_v1/trainer.py

    # Docker（挂载数据集和输出目录）
    docker run --gpus all --rm \
      -v /path/to/dataset:/data/dataset \
      -v /path/to/output:/data/output \
      seatbelt python rtdetr_seatbelt_detection_model_v1/trainer.py \
      --data /data/dataset/SeatbeltDetection/seatbelt_detection_data.yaml \
      --project /data/output
"""

import os
import argparse
from ultralytics import RTDETR

# 基于本文件位置解析默认路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)

_DEFAULT_WEIGHTS = os.path.join(_PROJECT_ROOT, "rtdetr_model_zoo", "rtdetr-l.pt")
_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "dataset", "SeatbeltDetection",
                              "seatbelt_detection_data.yaml")
_DEFAULT_PROJECT = os.path.join(_PROJECT_ROOT, "runs", "detect")
_PROJECT_NAME = "seatbelt_detection_train1"


def main():
    parser = argparse.ArgumentParser(description="RTDETR Seatbelt Detection Training v1")
    parser.add_argument("--weights", type=str, default=_DEFAULT_WEIGHTS,
                        help="预训练权重路径")
    parser.add_argument("--data", type=str, default=_DEFAULT_DATA,
                        help="数据集 yaml 路径")
    parser.add_argument("--project", type=str, default=_DEFAULT_PROJECT,
                        help="训练输出目录")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备，'cpu' 或 0,1,2...")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=800)
    args = parser.parse_args()

    # 权重文件存在性检查
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"预训练权重不存在: {args.weights}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置不存在: {args.data}")

    model = RTDETR(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        val=True,
        save=True,
        pretrained=True,
        patience=15,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mixup=0.25,
        copy_paste=0.2,
        mosaic=1.0,
        auto_augment="randaugment",
        cos_lr=True,
        close_mosaic=10,
        augment=False,
        profile=False,
        plots=True,
        workers=8,
        cache="ram",
        project=args.project,
        name=_PROJECT_NAME,
    )

    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"权重保存路径: {args.project}/{_PROJECT_NAME}/weights/")
    print("=" * 50)
    return results


if __name__ == "__main__":
    main()
