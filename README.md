# UAV Vehicle Seatbelt Detection

基于RT-DETR的无人机车辆安全带检测项目。

## 项目结构

```
├── dataset/                    # 数据集目录
│   ├── COCO2017/              # COCO 2017 数据集配置
│   ├── SeatbeltDetection/     # 安全带检测数据集配置
│   └── VisDrone2019/          # VisDrone 2019 数据集配置
├── rtdetr_model_zoo/          # RT-DETR 预训练模型
├── rtdetr_seatbelt_detection_model_v1/  # 安全带检测模型 v1
├── rtdetr_seatbelt_detection_model_v2/  # 安全带检测模型 v2
├── seatbelt_detection_v1/     # 检测程序 v1
├── seatbelt_detection_v2/     # 检测程序 v2
└── tools/                     # 辅助工具脚本
```

## 使用说明

### 1. 安装依赖

运行程序前，请先安装必要的依赖：

```bash
pip install ultralytics
```

### 2. 路径配置

运行程序前，请检查并确认所有路径配置正确。如果程序提示路径错误，请根据实际文件结构在代码中修正路径。

### 3. 模型权重替换

模型训练程序中使用的原始权重可以替换为 `rtdetr_model_zoo/` 目录中的其他权重文件：
- `.pt` 文件为预训练模型权重
- `.yaml` 文件为模型配置文件

两者均为 Ultralytics 架构支持的格式，可以根据需求自行替换。

**注意**：由于文件大小限制，`rtdetr-x.pt` 权重文件未包含在本仓库中。`rtdetr_model_zoo/` 目录仅提供 `rtdetr-x` 的 yaml 格式架构文件。如果需要使用其 `.pt` 实体权重进行推理或训练，请访问 [Ultralytics 官方网站](https://github.com/ultralytics/ultralytics) 下载。

### 4. 数据集配置

#### 官方数据集
如果使用官方数据集（如 COCO2017、VisDrone2019），请按照各训练程序中的注释进行路径和配置替换。

#### 自定义数据集
如果使用自划分或自建数据集，请遵循以下要求：
- 数据集格式：YOLO 格式
- 存放位置：`dataset/` 目录下的对应子目录
- 配置文件：参考现有 `.yaml` 文件格式编写

## 引用

如果您在研究中使用了本项目，请引用以下论文：

```
@misc{lv2023detrs,
       title={DETRs Beat YOLOs on Real-time Object Detection},
       author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
       year={2023},
       eprint={2304.08069},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
}
```

## 许可证

本项目仅供研究和学习使用。