# seatbelt_detector.py
from ultralytics import RTDETR
import numpy as np
import cv2
import argparse
import os

# 加载模型
model = RTDETR("../rtdetr_seatbelt_detection_model_v2/seatbelt_detection_train2/weights/best.pt")

# 类别ID映射
CLASS_MAP = {
    0: "person-noseatbelt",
    1: "person-seatbelt",
    2: "seatbelt",
    3: "windshield"
}
WINDOW_CLASS_ID = 3
SEATBELT_CLASS_ID = 2
PERSON_CLASS_IDS = [0, 1]

# 新增置信度阈值和最大数量限制
WINDOW_CONF_THRESHOLD = 0.7  # 车窗置信度阈值
SEATBELT_CONF_THRESHOLD = 0.6  # 安全带置信度阈值
PERSON_CONF_THRESHOLD = 0.4  # 人员置信度阈值（便于后续二次验证）
MAX_WINDOWS = 5  # 最多取5个车窗
MAX_PERSONS = 10  # 最多取10个人员
MAX_SEATBELTS = 10  # 最多取10个安全带
# 适用于改进IOU的阈值
WINDOW_IOU_THRESHOLD = 0.6  # 人员与车窗IOU阈值
SEATBELT_IOU_THRESHOLD = 0.8  # 人员与安全带IOU阈值
# 如果使用传统IOU，则将上面改进IOU的阈值注释掉，使用传统IOU的阈值
# WINDOW_IOU_THRESHOLD = 0.15  # 人员与车窗IOU阈值
# SEATBELT_IOU_THRESHOLD = 0.10  # 人员与安全带IOU阈值

def calculate_iou(box1, box2):
    """计算-改进IOU：计算box1与box2的交集面积占box2面积的比例"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 使用box2的面积作为分母（类似is_inside_window逻辑）
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    if area2 == 0:
        return 0.0
    
    return inter_area / area2

# def calculate_iou(box1, box2):
#     """计算两个边界框的IoU"""
#     x1_1, y1_1, x2_1, y2_1 = box1
#     x1_2, y1_2, x2_2, y2_2 = box2
    
#     inter_x1 = max(x1_1, x1_2)
#     inter_y1 = max(y1_1, y1_2)
#     inter_x2 = min(x2_1, x2_2)
#     inter_y2 = min(y2_1, y2_2)
    
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
#     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union_area = area1 + area2 - inter_area
    
#     return inter_area / union_area if union_area > 0 else 0.0

def detect_single_frame(image):
    """
    对单帧图像进行安全带检测（融合版：多辆车支持 + 规则二次验证）
    
    Args:
        image: 输入的BGR图像
    
    Returns:
        dict: 包含检测结果的字典
            - 'frame': 处理后的图像
            - 'results': 检测框信息列表
    """
    # 关键优化：使用与训练一致的尺寸
    results = model.predict(
        image,
        imgsz=800,  # 与训练尺寸一致
        conf=0.4,   # 人员类别分段阈值
        iou=0.8,    # 适当降低IOU阈值
        device='cpu',
        classes=PERSON_CLASS_IDS + [WINDOW_CLASS_ID, SEATBELT_CLASS_ID]
    )
    
    result = results[0]
    boxes = result.boxes
    ids = boxes.id if boxes.id is not None else list(range(len(boxes)))
    
    # 优化1：提取置信度最高的车窗（最多5个）
    window_boxes = []
    for box in boxes:
        if int(box.cls) == WINDOW_CLASS_ID and box.conf.item() >= WINDOW_CONF_THRESHOLD:
            window_boxes.append((box.xyxy[0].tolist(), box.conf.item()))
    
    # 按置信度降序排序，取前MAX_WINDOWS个
    window_boxes.sort(key=lambda x: x[1], reverse=True)
    window_bboxes = [box[0] for box in window_boxes[:MAX_WINDOWS]]
    
    # 优化2：提取置信度最高的安全带（最多10个）
    seatbelt_boxes = []
    for box in boxes:
        if int(box.cls) == SEATBELT_CLASS_ID and box.conf.item() >= SEATBELT_CONF_THRESHOLD:
            seatbelt_boxes.append((box.xyxy[0].tolist(), box.conf.item()))
    
    # 按置信度降序排序，取前MAX_SEATBELTS个
    seatbelt_boxes.sort(key=lambda x: x[1], reverse=True)
    seatbelt_bboxes = [box[0] for box in seatbelt_boxes[:MAX_SEATBELTS]]
    
    # 优化3：拆分人员框为高/中/低置信度
    high_conf_person_boxes = []  # >0.7 直接输出
    mid_conf_person_boxes = []   # 0.4-0.7 二次验证
    # 存储人员框及其所属类别、置信度，方便后续处理
    for box, obj_id in zip(boxes, ids):
        cls_id = int(box.cls)
        conf = box.conf.item()  # 获取置信度值
        if cls_id in PERSON_CLASS_IDS:
            if conf > 0.7:
                high_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
            elif 0.4 <= conf <= 0.7:
                mid_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
            # <0.4 直接丢弃，不加入任何列表
    
    # 优化4：仅对中置信度框应用修正逻辑（处理未系安全带和已系安全带人员）
    revised_cls = {}
    
    for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
        if int(obj_id) in revised_cls:
            continue  # 已修正
        
        # 计算人员与所有车窗的IOU最大值
        best_window_iou = 0
        for window_bbox in window_bboxes:
            # 正确参数顺序：车窗框在前，人员框在后
            iou = calculate_iou(window_bbox, person_bbox)
            if iou > best_window_iou:
                best_window_iou = iou
        
        # 只有在车窗内的人员才进行安全带状态修正
        if best_window_iou >= WINDOW_IOU_THRESHOLD:
            # 计算人员与所有安全带的IOU最大值
            best_seatbelt_iou = 0
            for sb in seatbelt_bboxes:
                # 正确参数顺序：人员框在前，安全带框在后
                iou = calculate_iou(person_bbox, sb)
                if iou > best_seatbelt_iou:
                    best_seatbelt_iou = iou
            
            # 修正逻辑：根据安全带IOU修正人员状态
            if cls_id == 0:  # 未系安全带人员
                if best_seatbelt_iou >= SEATBELT_IOU_THRESHOLD:
                    revised_cls[int(obj_id)] = 1  # 修正为已系安全带
            elif cls_id == 1:  # 已系安全带人员
                if best_seatbelt_iou < SEATBELT_IOU_THRESHOLD:
                    revised_cls[int(obj_id)] = 0  # 修正为未系安全带
    
    # 处理检测结果（融合版：区分高/中置信度逻辑）
    frame = image.copy()
    all_results = []  # 存储最终结果
    
    # 处理高置信度人员框：直接输出，不校正、不判断车窗
    for person_bbox, obj_id, cls_id, conf in high_conf_person_boxes:
        x1, y1, x2, y2 = map(int, person_bbox)
        obj_id = int(obj_id)
        
        # 高置信度直接输出类别，不判断车窗、不校正
        if cls_id == 0:
            label = f"Unbelted"
            box_color = (0, 0, 255)
            text_color = (0, 0, 255)
        else:
            label = f"Belted"
            box_color = (0, 255, 0)
            text_color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # 记录高置信度结果（is_inside设为True，因未判断车窗）
        all_results.append({
            'bbox': person_bbox,
            'cls': cls_id,
            'id': obj_id,
            'is_inside': True,
            'conf': conf
        })
    
    # 处理中置信度人员框：先判断车窗，再应用安全带状态校正
    for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
        x1, y1, x2, y2 = map(int, person_bbox)
        obj_id = int(obj_id)
        
        # 先应用安全带状态修正（仅车窗内的框会被修正）
        if int(obj_id) in revised_cls:
            cls_id = revised_cls[int(obj_id)]
        
        # 计算人员与所有车窗的IOU最大值，判断是否在车内
        best_window_iou = 0
        best_window_bbox = None
        for window_bbox in window_bboxes:
            # 正确参数顺序：车窗框在前，人员框在后
            iou = calculate_iou(window_bbox, person_bbox)
            if iou > best_window_iou:
                best_window_iou = iou
                print(f"Best window IOU: {best_window_iou}")
                best_window_bbox = window_bbox
        
        is_inside = best_window_iou >= WINDOW_IOU_THRESHOLD
        
        if is_inside:
            if cls_id == 0:
                label = f"Unbelted"
                box_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                label = f"Belted"
                box_color = (0, 255, 0)
                text_color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        else:
            label = f"Outside"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 记录中置信度结果
        all_results.append({
            'bbox': person_bbox,
            'cls': cls_id,
            'id': obj_id,
            'is_inside': is_inside,
            'conf': conf
        })
    
    # 绘制所有检测到的车窗边界
    for i, window_bbox in enumerate(window_bboxes):
        wx1, wy1, wx2, wy2 = map(int, window_bbox)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(frame, f"Windshield_{i+1}", (wx1, wy1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return {
        'frame': frame,
        'results': all_results
    }

if __name__ == "__main__":
    """
    单张图片推理入口
    
    使用方法:
    python seatbelt_detector.py --image_path /path/to/input/image.jpg
    
    注意: 请将 --image_path 参数替换为实际图片路径
    """
    parser = argparse.ArgumentParser(description='Seatbelt Detection on Single Image')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the input image file (e.g., /home/user/image.jpg)')
    args = parser.parse_args()
    
    # 检查输入图片是否存在
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found at: {args.image_path}")
    
    # 读取图片
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Failed to read image from: {args.image_path}")
    
    # 执行检测
    result = detect_single_frame(image)
    
    # 保存结果（不再显示窗口）
    output_path = os.path.splitext(args.image_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, result['frame'])
    print(f"Result saved to: {output_path}")