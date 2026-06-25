"""
RTDETR 安全带检测模型训练入口 v2

原 train_v2.py 的路径问题修复版：使用 __file__ 解析相对路径，不依赖 CWD。
Docker 环境下同样可用（只需挂载 dataset 目录并传参覆盖默认路径）。

Usage:
    # 本地
    python rtdetr_seatbelt_detection_model_v2/trainer.py

    # Docker（挂载数据集和输出目录）
    docker run --gpus all --rm \
      -v /path/to/dataset:/data/dataset \
      -v /path/to/output:/data/output \
      seatbelt python rtdetr_seatbelt_detection_model_v2/trainer.py \
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
_PROJECT_NAME = "seatbelt_detection_train2"


def main():
    parser = argparse.ArgumentParser(description="RTDETR Seatbelt Detection Training v2")
    parser.add_argument("--weights", type=str, default=_DEFAULT_WEIGHTS,
                        help="预训练权重路径")
    parser.add_argument("--data", type=str, default=_DEFAULT_DATA,
                        help="数据集 yaml 路径")
    parser.add_argument("--project", type=str, default=_DEFAULT_PROJECT,
                        help="训练输出目录")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备，'cpu' 或 0,1,2...")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
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
        workers=8,
        pretrained=True,
        patience=20,
        seed=0,
        deterministic=True,

        # 优化器
        optimizer="AdamW",
        lr0=0.00005,
        lrf=0.005,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 损失函数权重
        cls=0.7,
        dfl=1.8,
        box=7.5,

        # 数据增强
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.3,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.1,
        auto_augment="randaugment",
        erasing=0.4,

        # 训练策略
        cos_lr=True,
        close_mosaic=15,
        amp=True,
        cache="disk",
        rect=False,
        project=args.project,
        name=_PROJECT_NAME,
        exist_ok=False,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"权重保存路径: {args.project}/{_PROJECT_NAME}/weights/")
    print("=" * 50)
    return results


if __name__ == "__main__":
    main()
