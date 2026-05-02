from ultralytics.data.utils import download

from ultralytics import RTDETR

model = RTDETR("../rtdetr_model_zoo/rtdetr-l.pt")

model.val(
    # 如果采用自划分COCO2017数据集，注释掉data="coco128.yaml"，否则注释掉data="../dataset/COCO2017/coco2017_data.yaml"
    data="../dataset/COCO2017/coco2017_data.yaml", 
    # Ultralytics官方自带coco数据集，有多种不同划分，可以根据需要，参照官方文档修改
    # data="coco128.yaml"   
    imgsz=640, 
    device=0,
    batch=16,
    conf=0.001, 
)