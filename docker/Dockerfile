# ============================================================
# UAV Vehicle Seatbelt Detection — Dockerfile
# 基于 RT-DETR (Ultralytics) 的无人机车辆安全带检测
#
# 使用方式：根据需要取消对应段的注释，注释掉另一段
# ============================================================

# ==================== 方案一：CPU 推理 ====================
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    numpy \
    pyyaml \
    Pillow

# ==================== 方案二：GPU 训练 / 推理 ====================
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
#
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1
# ENV PATH="/usr/local/bin:${PATH}"
#
# WORKDIR /app
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3.10-distutils \
#     python3-pip \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/* \
#     && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
#     && python -m pip install --no-cache-dir --upgrade pip
#
# RUN pip install --no-cache-dir \
#     torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
#     && pip install --no-cache-dir \
#     ultralytics \
#     opencv-python \
#     numpy \
#     pyyaml \
#     Pillow

# ============================================================

# 复制整个工程
COPY . /app/

# 默认入口
CMD ["python", "-c", "print('Seatbelt Detection Container Ready.')"]
