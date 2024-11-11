from ultralytics import YOLO

# Load YOLO model and start training
model = YOLO('yolov8n.pt')  # Start with the YOLOv8 model
model.train(data='./rc_car_dataset.yaml', epochs=100, imgsz=640)
