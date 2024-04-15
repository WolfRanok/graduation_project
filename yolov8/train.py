from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 加载训练的模型

# 模型训练
results = model.train(data='coco128.yaml', epochs=300, imgsz=640)