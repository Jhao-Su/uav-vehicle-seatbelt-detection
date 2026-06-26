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
conda create -n seatbelt python=3.8 -y
conda activate seatbelt

# 安装依赖
pip install ultralytics opencv-python numpy pyyaml Pillow
```

#### Docker 环境

项目提供了 Dockerfile（位于 `docker/` 目录），支持 CPU 推理和 GPU 训练两种方案。使用时打开 `docker/Dockerfile`，根据需要**取消对应方案的注释、注释掉另一方案**即可：

```bash
# 当前默认启用方案一（CPU），如需 GPU 则：
#   1. 注释掉方案一的 FROM / ENV / RUN 等行
#   2. 取消方案二各行的注释
#   3. 构建并运行
docker build -t seatbelt -f docker/Dockerfile .
```

**使用示例**

```bash
# 单张图片检测
docker run --rm -v /path/to/images:/data seatbelt python -c "
from seatbelt_detection_v2.detector import SeatbeltDetector; import cv2
d = SeatbeltDetector({}); r = d.recognize_image(cv2.imread('/data/test.jpg'))
cv2.imwrite('/data/test_result.jpg', r['annotated_image']); d.clean()"

# 视频检测
docker run --rm -v /path/to/videos:/data seatbelt python -c "
from seatbelt_detection_v2.detector import SeatbeltDetector
d = SeatbeltDetector({})
d.process_video('/data/test.mp4', '/data/output')
d.clean()"

# 模型训练（需先切换为 GPU 方案）
docker run --gpus all --rm -v /path/to/dataset:/app/dataset \
  seatbelt python rtdetr_seatbelt_detection_model_v2/trainer.py \
  --data /data/dataset/SeatbeltDetection/seatbelt_detection_data.yaml \
  --project /data/output
```

> **注意**：Docker 镜像已内置模型权重文件（`.pt`）。数据集和测试数据通过 `-v` 挂载使用，避免镜像体积过大。
>
> **GPU 方案说明**：由于本项目开发条件限制，Dockerfile 的 GPU 方案暂未经实际测试。如需 GPU 训练或推理，建议优先使用 Conda 环境部署。欢迎有条件的使用者测试并反馈。

#### CPU / GPU 切换

各程序中使用 `device='cpu'` 或 `device=0` 控制推理/训练设备。运行前根据需要注释掉不需要的行即可：

```python
# CPU 推理
device='cpu',

# GPU 推理（取消注释即可）
# device=0,
```

涉及的文件：

| 文件 | 所在行附近 |
|------|-----------|
| `seatbelt_detection_v1/detector.py` | `config["device"]` 配置项 |
| `seatbelt_detection_v2/detector.py` | `config["device"]` 配置项 |
| `rtdetr_seatbelt_detection_model_v1/trainer.py` | `model.train(...)` 调用中 |
| `rtdetr_seatbelt_detection_model_v2/trainer.py` | `model.train(...)` 调用中 |

### 2. 检测程序使用

每个版本的目录下仅有一个入口文件 `detector.py`，基于 `SeatbeltDetector` 类提供统一的单帧检测和视频处理能力。

#### 单帧图片检测

对单张图片进行推理，返回标注后的图像及各车辆内人员的结构化检测结果。

```python
import cv2
from seatbelt_detection_v2.detector import SeatbeltDetector

detector = SeatbeltDetector({})                       # 使用默认模型路径
image = cv2.imread("/path/to/image.jpg")
result = detector.recognize_image(image)
cv2.imwrite("/path/to/output.jpg", result["annotated_image"])
detector.clean()
```

返回字段详见 [接口说明](#接口说明)。v1 与 v2 接口完全一致，替换 import 路径即可切换。

#### 视频处理

对视频文件逐帧检测，输出带标注框的结果视频，支持跳帧加速。

```python
from seatbelt_detection_v2.detector import SeatbeltDetector

detector = SeatbeltDetector({})
detector.process_video("/path/to/input.mp4", "/path/to/output.mp4", skip_frames=0)
detector.clean()
```

#### 系统集成

作为算法模块嵌入无人机巡检等外部系统，通过 config dict 配置模型路径、设备、阈值等所有参数。

```python
from seatbelt_detection_v2.detector import SeatbeltDetector

config = {
    "model_path": "/opt/models/seatbelt/best.pt",  # 部署时使用绝对路径
    "device": 0,                                    # GPU 推理
    "conf_threshold": 0.4,
}
detector = SeatbeltDetector(config)
result = detector.recognize_image(image)            # result 为结构化 dict
detector.clean()
```

#### 接口说明

`recognize_image()` 与 `process_video()` 的返回值结构如下。

```python
{
    "annotated_image": np.ndarray,   # 带标注框的结果图像
    "vehicles": [{                   # 按车辆分组
        "vehicle_id": 0,
        "window_bbox": [x1, y1, x2, y2],
        "window_confidence": 0.917,
        "persons": [{
            "bbox": [x, y, w, h],             # xywh 格式
            "confidence": 0.911,
            "class_name": "person-noseatbelt", # "person-noseatbelt" / "person-seatbelt"
            "object_id": 1,
            "in_window": True,
        }],
    }],
    "image_info": {"width": 1920, "height": 1080},
    "has_target": True,
    "vehicle_count": 1,
}
```

`process_video()` 返回：

```python
{
    "status": "success",
    "output_video_path": "/path/to/output.mp4",
    "total_frames": 300,
    "processed_frames": 300,
}
```

> **v1 与 v2 的区别**：v1 对所有人员统一进行车窗判位和安全带 IOU 二次验证（`conf_threshold` 默认 0.6）；v2 采用分段置信度策略——高置信度（>0.7）直接输出、中置信度（0.4-0.7）经二次验证后输出。调用方式完全相同，替换 import 中 `v1` / `v2` 即可切换。

### 3. 路径配置

运行程序前，请检查并确认所有路径配置正确。如果程序提示路径错误，请根据实际文件结构在代码中修正路径。

### 4. 模型权重替换

模型训练程序中使用的原始权重可以替换为 `rtdetr_model_zoo/` 目录中的其他权重文件：
- `.pt` 文件为预训练模型权重
- `.yaml` 文件为模型配置文件

两者均为 Ultralytics 架构支持的格式，可以根据需求自行替换。

**注意**：由于文件大小限制，`rtdetr-x.pt` 权重文件未包含在本仓库中。`rtdetr_model_zoo/` 目录仅提供 `rtdetr-x` 的 yaml 格式架构文件。如果需要使用其 `.pt` 实体权重进行推理或训练，请访问 [Ultralytics 官方网站](https://github.com/ultralytics/ultralytics) 下载。

### 5. 数据集配置

#### 官方数据集
如果使用官方数据集（如 COCO2017、VisDrone2019），请按照各训练程序中的注释进行路径和配置替换。

#### 自定义数据集
如果使用自划分或自建数据集，请遵循以下要求：
- 数据集格式：YOLO 格式
- 存放位置：`dataset/` 目录下的对应子目录
- 配置文件：参考现有 `.yaml` 文件格式编写

### 6. 辅助工具

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