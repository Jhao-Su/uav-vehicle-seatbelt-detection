"""
使用说明
-------
本脚本用于从视频文件中逐帧提取图片并保存为 JPG 格式。

使用方法：
    1. 修改 __main__ 中的 video_file 变量为你的视频文件路径
    2. 修改 output_directory 变量为你想要保存帧图片的目录
    3. 运行脚本：
       python cut_video_frames.py

输出：
    帧图片以 frame_000001.jpg, frame_000002.jpg ... 的格式保存在指定目录中。

依赖：
    opencv-python (cv2)
"""

import cv2
import os

def extract_frames(video_path, output_dir):
    """
    从视频文件中提取每一帧并保存为JPG图片
    
    参数:
    video_path: 视频文件路径
    output_dir: 帧图片保存目录
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    frame_count = 0
    
    try:
        # 循环读取视频帧
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            # 如果读取失败，说明已经到视频末尾
            if not ret:
                break
            
            # 构建输出文件名
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            
            # 保存帧为JPG图片
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            
            # 每100帧打印一次进度
            if frame_count % 100 == 0:
                print(f"已提取 {frame_count} 帧")
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    
    finally:
        # 释放资源
        cap.release()
        print(f"提取完成，共提取 {frame_count} 帧")

if __name__ == "__main__":
    # 视频文件路径
    video_file = "test5.mp4"
    # 输出目录
    output_directory = "video_cut"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_file):
        print(f"视频文件不存在: {video_file}")
    else:
        extract_frames(video_file, output_directory)