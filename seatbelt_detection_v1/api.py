"""
安全带检测模块 v1 — 系统集成入口

不修改原始 seatbelt_detector.py，通过包装层解决模块导入时的路径依赖问题。
适合作为无人机巡检系统的算法子模块调用。

Usage:
    from seatbelt_detection_v1.api import detect_single_frame, process_video

    # 单帧检测（首次调用时自动加载模型）
    result = detect_single_frame(image)

    # 视频处理
    process_video("/path/to/video.mp4", "/path/to/output")
"""

import os
import sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)

# 确保项目根在 sys.path 中，以便 importlib 能发现子模块
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_detector = None


def _load_detector():
    """懒加载：首次调用时导入原 seatbelt_detector，此时切换 CWD 保证相对模型路径正确"""
    global _detector
    if _detector is not None:
        return _detector

    import importlib

    _saved_cwd = os.getcwd()
    # 切换到本模块所在目录，使原 seatbelt_detector.py 中的 "../rtdetr_..." 路径正确解析
    os.chdir(_BASE_DIR)
    try:
        _detector = importlib.import_module("seatbelt_detection_v1.seatbelt_detector")
    finally:
        os.chdir(_saved_cwd)

    return _detector


def detect_single_frame(image):
    """
    对单帧图像进行安全带检测。

    Args:
        image: BGR 格式的 numpy 数组（cv2 读取结果）

    Returns:
        dict: {"frame": 绘制结果图像, "results": [检测框信息列表]}
    """
    return _load_detector().detect_single_frame(image)


def process_video(video_path, output_dir, skip_frames=0):
    """
    对视频文件进行逐帧安全带检测。

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录（结果保存为 result_output.mp4）
        skip_frames: 跳帧数，0 表示不跳过
    """
    import cv2

    # 先触发模型加载（若尚未加载）
    _load_detector()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result_output.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap = cv2.VideoCapture(video_path)
    processed = 0

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Output: {output_path}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            continue

        result = detect_single_frame(frame)
        out.write(result["frame"])

        if processed % 10 == 0:
            print(f"Progress: {processed}/{total_frames} ({processed / total_frames:.1%})")

        processed += 1

    cap.release()
    out.release()
    print(f"Done. Result: {output_path}")
