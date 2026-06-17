# ============================================================
# UAV Vehicle Seatbelt Detection — Dockerfile
# 基于 RT-DETR (Ultralytics) 的无人机车辆安全带检测
#
# 多阶段构建：
#   - cpu 阶段：轻量推理镜像（仅 CPU）
#   - gpu 阶段：训练 + 推理镜像（CUDA 12.1）
# ============================================================

# ==================== CPU 推理阶段 ====================
FROM python:3.10-slim AS cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装 OpenCV 所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    numpy \
    pyyaml \
    Pillow

# 复制整个工程
COPY . /app/

# 默认入口：打印就绪信息
CMD ["python", "-c", "print('Seatbelt Detection Container (CPU) Ready.')"]


# ==================== GPU 训练 / 推理阶段 ====================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

# 安装 Python 3.10 与 OpenCV 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --no-cache-dir --upgrade pip

# 先装 PyTorch + CUDA，再装 ultralytics（避免重复下载 torch）
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    numpy \
    pyyaml \
    Pillow

# 复制整个工程
COPY . /app/

# 默认入口
CMD ["python", "-c", "print('Seatbelt Detection Container (GPU) Ready.')"]
