"""
安全带检测模块 v2 — 系统集成类封装

保留原有 seatbelt_detector.py 的检测算法与绘制风格不变，
通过类封装 + 配置驱动 + 结构化返回值，适配系统集成场景。

Usage:
    from seatbelt_detection_v2.detector import SeatbeltDetector

    config = {"model_path": "path/to/best.pt"}
    detector = SeatbeltDetector(config)
    result = detector.recognize_image(image)
    detector.clean()
"""

import os
import logging
import gc
import cv2
import numpy as np
from ultralytics import RTDETR

logger = logging.getLogger(__name__)

# 根据本文件位置解析默认模型路径（与 seatbelt_detector.py 一致）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(_BASE_DIR),
    "rtdetr_seatbelt_detection_model_v2",
    "seatbelt_detection_train2",
    "weights",
    "best.pt",
)


class SeatbeltDetector:
    """安全带检测器，基于 RTDETR + 分段置信度策略 + 车窗和安全带 IOU 二次验证"""

    # ---------- 与 seatbelt_detector.py 完全一致的常量 ----------
    CLASS_MAP = {
        0: "person-noseatbelt",
        1: "person-seatbelt",
        2: "seatbelt",
        3: "windshield",
    }
    WINDOW_CLASS_ID = 3
    SEATBELT_CLASS_ID = 2
    PERSON_CLASS_IDS = [0, 1]

    def __init__(self, config):
        """
        Args:
            config: dict，支持以下键:
                - model_path:  模型权重路径（必填，默认指向 v2 训练权重）
                - conf_threshold: 模型推理置信度 (默认 0.4)
                - iou_threshold:  模型推理 IOU (默认 0.8)
                - imgsz:          推理尺寸 (默认 800)
                - device:         'cpu' 或 0 (默认 'cpu')
                - window_conf:    车窗筛选置信度 (默认 0.7)
                - seatbelt_conf:  安全带筛选置信度 (默认 0.6)
                - person_conf:    人员最低置信度 (默认 0.4)
                - max_windows:    最多车窗数 (默认 5)
                - max_persons:    最多人员数 (默认 10)
                - max_seatbelts:  最多安全带数 (默认 10)
                - window_iou_thr: 人员-车窗 IOU 阈值 (默认 0.6)
                - seatbelt_iou_thr: 人员-安全带 IOU 阈值 (默认 0.8)
        """
        # 模型路径
        self.model_path = config.get("model_path", _DEFAULT_MODEL)

        # 推理参数
        self.conf_threshold = config.get("conf_threshold", 0.4)
        self.iou_threshold = config.get("iou_threshold", 0.8)
        self.imgsz = config.get("imgsz", 800)
        self.device = config.get("device", "cpu")

        # 后处理阈值 — 与 seatbelt_detector.py 一致
        self.window_conf = config.get("window_conf", 0.7)
        self.seatbelt_conf = config.get("seatbelt_conf", 0.6)
        self.person_conf = config.get("person_conf", 0.4)
        self.max_windows = config.get("max_windows", 5)
        self.max_persons = config.get("max_persons", 10)
        self.max_seatbelts = config.get("max_seatbelts", 10)
        self.window_iou_thr = config.get("window_iou_thr", 0.6)
        self.seatbelt_iou_thr = config.get("seatbelt_iou_thr", 0.8)

        # 加载模型
        logger.info("加载模型: %s", self.model_path)
        self.model = RTDETR(self.model_path)
        logger.info("模型加载成功")

    def clean(self):
        """释放模型及 GPU 资源"""
        logger.info("清理资源")
        if hasattr(self, "model"):
            del self.model
        gc.collect()

    # ---------- 与 seatbelt_detector.py 完全一致的 IOU ----------
    @staticmethod
    def _iou(box1, box2):
        """改进 IOU：box1 与 box2 交集面积 / box2 面积"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / area2 if area2 > 0 else 0.0

    # ---------- 核心检测（与 seatbelt_detector.py v2 算法完全一致）----------
    def _process_image(self, image):
        if image is None:
            return {"error": "无效的图像数据", "has_target": False}

        h, w = image.shape[:2]
        frame = image.copy()

        # --- 模型推理（参数与 seatbelt_detector.py 完全一致）---
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.PERSON_CLASS_IDS + [self.WINDOW_CLASS_ID, self.SEATBELT_CLASS_ID],
            verbose=False,
            stream=True,
        )
        result = next(results)
        boxes = result.boxes
        ids = boxes.id if boxes.id is not None else list(range(len(boxes)))

        # --- 提取车窗（最多 max_windows 个，按置信度排序）---
        window_boxes = []
        for box in boxes:
            if int(box.cls) == self.WINDOW_CLASS_ID and box.conf.item() >= self.window_conf:
                window_boxes.append((box.xyxy[0].tolist(), box.conf.item()))
        window_boxes.sort(key=lambda x: x[1], reverse=True)
        window_boxes = window_boxes[:self.max_windows]

        # --- 提取安全带（最多 max_seatbelts 个，按置信度排序）---
        sb_boxes = []
        for box in boxes:
            if int(box.cls) == self.SEATBELT_CLASS_ID and box.conf.item() >= self.seatbelt_conf:
                sb_boxes.append((box.xyxy[0].tolist(), box.conf.item()))
        sb_boxes.sort(key=lambda x: x[1], reverse=True)
        seatbelt_bboxes = [b[0] for b in sb_boxes[:self.max_seatbelts]]

        # --- 人员拆分：高置信度(>0.7) / 中置信度(0.4-0.7)（v2 策略）---
        high_conf = []
        mid_conf = []
        for box, obj_id in zip(boxes, ids):
            cls_id = int(box.cls)
            if cls_id not in self.PERSON_CLASS_IDS:
                continue
            conf = box.conf.item()
            if conf < self.person_conf:
                continue
            item = (box.xyxy[0].tolist(), int(obj_id), cls_id, conf)
            if conf > 0.7:
                high_conf.append(item)
            else:
                mid_conf.append(item)

        # --- 安全带修正（仅对中置信度、在车窗内的人员）---
        revised_cls = {}
        for pbox, oid, cls_id, conf in mid_conf:
            best_win_iou = max(
                (self._iou(wb, pbox) for wb, _ in window_boxes), default=0.0
            )
            if best_win_iou < self.window_iou_thr:
                continue
            best_sb_iou = max(
                (self._iou(pbox, sb) for sb in seatbelt_bboxes), default=0.0
            )
            if cls_id == 0 and best_sb_iou >= self.seatbelt_iou_thr:
                revised_cls[oid] = 1
            elif cls_id == 1 and best_sb_iou < self.seatbelt_iou_thr:
                revised_cls[oid] = 0

        # --- 绘制与结果收集 — 保持原有绘制风格 ---
        all_results = []
        # 记录人员归属哪辆车（用于结构化输出）
        person_vehicle_map = {}

        # 1) 高置信度人员：直接输出，不判断车窗
        for pbox, oid, cls_id, conf in high_conf:
            self._draw_person(frame, pbox, oid, cls_id, is_inside=True)
            all_results.append({"bbox": pbox, "cls": cls_id, "id": oid, "is_inside": True, "conf": conf})
            # 高置信度归属到匹配度最高的车窗
            best_vid = 0
            best_iou = 0.0
            for vid, (wb, _) in enumerate(window_boxes):
                iou_val = self._iou(wb, pbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_vid = vid
            person_vehicle_map[oid] = best_vid

        # 2) 中置信度人员：判断车窗 + 安全带修正
        for pbox, oid, cls_id, conf in mid_conf:
            if oid in revised_cls:
                cls_id = revised_cls[oid]

            best_win_iou = 0.0
            best_vid = 0
            for vid, (wb, _) in enumerate(window_boxes):
                iou_val = self._iou(wb, pbox)
                if iou_val > best_win_iou:
                    best_win_iou = iou_val
                    best_vid = vid
            is_inside = best_win_iou >= self.window_iou_thr

            self._draw_person(frame, pbox, oid, cls_id, is_inside)
            all_results.append({"bbox": pbox, "cls": cls_id, "id": oid, "is_inside": is_inside, "conf": conf})
            if is_inside:
                person_vehicle_map[oid] = best_vid

        # 3) 绘制车窗 — 与原有保持一致
        for i, (wbox, _) in enumerate(window_boxes):
            wx1, wy1, wx2, wy2 = map(int, wbox)
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
            cv2.putText(frame, f"Windshield_{i+1}", (wx1, wy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- 组装结构化返回（按车辆分组，适配系统集成）---
        vehicles = []
        for vid, (wbox, wconf) in enumerate(window_boxes):
            persons_in_vehicle = []
            for r in all_results:
                if person_vehicle_map.get(r["id"]) == vid and r["is_inside"]:
                    x1, y1, x2, y2 = map(int, r["bbox"])
                    persons_in_vehicle.append({
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": r["conf"],
                        "class_name": self.CLASS_MAP[r["cls"]],
                        "object_id": r["id"],
                        "in_window": True,
                    })
            vehicles.append({
                "vehicle_id": vid,
                "window_bbox": wbox,
                "window_confidence": wconf,
                "persons": persons_in_vehicle,
            })

        has_target = any(len(v["persons"]) > 0 for v in vehicles)

        return {
            "annotated_image": frame,
            "vehicles": vehicles,
            "image_info": {"width": w, "height": h},
            "has_target": has_target,
            "vehicle_count": len(window_boxes),
        }

    # ---------- 绘制（与原有完全一致）----------
    def _draw_person(self, frame, pbox, oid, cls_id, is_inside):
        x1, y1, x2, y2 = map(int, pbox)
        if is_inside:
            if cls_id == 0:
                label = "Unbelted"
                color = (0, 0, 255)
            else:
                label = "Belted"
                color = (0, 255, 0)
        else:
            label = "Outside"
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- 对外接口 ----------
    def recognize_image(self, image):
        """对 OpenCV 图像执行安全带检测，返回结构化结果"""
        logger.info("处理 OpenCV 图像")
        try:
            return self._process_image(image)
        except Exception as e:
            logger.error("检测失败: %s", e)
            return {"error": str(e), "has_target": False}

    def process_video(self, input_video_path, output_video_path=None, skip_frames=0):
        """处理视频文件，逐帧检测并输出标注视频"""
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        if not output_video_path:
            name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_path = os.path.join(
                os.path.dirname(input_video_path), f"{name}_result.mp4"
            )

        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        cap = cv2.VideoCapture(input_video_path)
        processed = 0
        idx = 0

        logger.info("处理视频: %s (%d 帧)", input_video_path, total)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if skip_frames > 0 and idx % (skip_frames + 1) != 0:
                continue
            result = self.recognize_image(frame)
            if "annotated_image" in result:
                out.write(result["annotated_image"])
            processed += 1
            if processed % 10 == 0:
                logger.info("进度: %d/%d (%.1f%%)", processed, total,
                            processed / total * 100)
        cap.release()
        out.release()
        logger.info("完成: %s", output_video_path)

        return {
            "status": "success",
            "output_video_path": output_video_path,
            "total_frames": total,
            "processed_frames": processed,
        }


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    detector = SeatbeltDetector({"model_path": _DEFAULT_MODEL, "device": "cpu"})
    try:
        # —— 单帧检测 ——
        # import argparse
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--image_path", type=str, required=True)
        # args = parser.parse_args()
        # image = cv2.imread(args.image_path)
        # result = detector.recognize_image(image)
        # output = args.image_path.replace(".jpg", "_detector_result.jpg")
        # cv2.imwrite(output, result["annotated_image"])
        # print("saved:", output)

        # —— 视频检测 ——
        # detector.process_video("/path/to/video.mp4")

        print("SeatbeltDetector v2 已就绪。")
    finally:
        detector.clean()
