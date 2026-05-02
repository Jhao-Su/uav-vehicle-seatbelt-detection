import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import RTDETR
import torch
import gc

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 加载预训练模型权重
model = RTDETR("../rtdetr_model_resnet50/rtdetr-l.pt")

model.train(
    # 数据输入和输出路径
    # 如果采用单独下载的VisDrone2019数据集，注释掉data="VisDrone2019.yaml"，否则注释掉data="../dataset/VisDrone2019/visdrone2019_data.yaml"
    data="../dataset/VisDrone2019/visdrone2019_data.yaml",  
    # Ultralytics官方自带VisDrone数据集，有多种不同划分，可以根据需要，参照官方文档修改
    # data="VisDrone2019.yaml"   
    project="../rtdetr_visdrone_train_val/train_val_result",

    # 训练基础配置
    epochs=150,             #  增加训练轮数
    val=True,
    patience=30,            #  增加耐心值
    resume=False,

    # 图像配置
    imgsz=960,              
    batch=8,                

    # 计算资源设置
    device=0,
    workers=8,
    cache='disk',
    deterministic=False,

    #  优化策略 - 关键优化
    optimizer="AdamW",
    lr0=0.00005,            # 降低初始学习率
    lrf=0.05,               # 提高最终学习率比例
    cos_lr=True,
    weight_decay=0.0005,
    warmup_epochs=10,       #  增加warmup
    warmup_momentum=0.8,
    
    #  损失权重 - 减少漏检
    box=5.0,                #  降低box权重
    cls=1.0,                #  增加cls权重
    iou=0.5,
    
    #  数据增强 - 针对小目标
    amp=True,
    close_mosaic=10,
    end2end=True,           #  
    scale=0.8,              #  减少缩放幅度
    translate=0.05,         #  减少平移
    mixup=0.1,              #  启用mixup
    perspective=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    
    #  检测配置
    conf=0.25,              #  降低置信度阈值
)