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

### 1. 环境配置

#### Conda 环境（推荐）

使用 Conda 创建隔离的 Python 环境并安装依赖：

```bash
# 创建并激活环境
conda create -n seatbelt python=3.10 -y
conda activate seatbelt

# 安装依赖
pip install ultralytics opencv-python numpy pyyaml Pillow
```

#### Docker 环境

项目提供了开箱即用的 Dockerfile（位于 `docker/` 目录），支持 CPU 推理和 GPU 训练两种场景。

**构建镜像**

```bash
# CPU 推理镜像（轻量，仅需 CPU）
docker build -t seatbelt:cpu --target cpu -f docker/Dockerfile .

# GPU 训练 / 推理镜像（需要 NVIDIA GPU + CUDA 驱动）
docker build -t seatbelt:gpu --target gpu -f docker/Dockerfile .
```

**使用示例**

```bash
# 单张图片检测（CPU）
docker run --rm -v /path/to/images:/data \
  seatbelt:cpu python seatbelt_detection_v2/seatbelt_detector.py --image_path /data/test.jpg

# 视频检测（CPU）
docker run --rm -v /path/to/videos:/data \
  seatbelt:cpu python seatbelt_detection_v2/video_process.py \
  --video_path /data/test.mp4 --output_dir /data/output

# 模型训练（GPU，需要 --gpus all）
docker run --gpus all --rm -v /path/to/dataset:/app/dataset \
  seatbelt:gpu python rtdetr_seatbelt_detection_model_v2/train_v2.py
```

> **说明**：Docker 镜像已内置模型权重文件（`.pt`）。数据集和测试数据通过 `-v` 挂载使用，避免镜像体积过大。更多细节请查看 `docker/Dockerfile` 中的注释。

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

### 5. 辅助工具

`tools/` 目录下提供了一系列辅助脚本，用于数据预处理和格式转换等操作。使用前请先打开对应脚本，参照脚本开头的 `''' '''` 使用说明进行配置和运行。

| 脚本 | 功能 |
|------|------|
| `cut_video_frames.py` | 从视频文件中逐帧提取图片 |
| `clean_empty_data.py` | 筛选有效的"图片-标注"对 |
| `clean_segment.py` | 清洗 YOLO 标注文件中的多余分割数据 |
| `data_visualize.py` | 在图片上绘制 YOLO 标注框，用于可视化检查 |
| `png2jpg.py` | 将 PNG 图片批量转换为 JPG 格式 |

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