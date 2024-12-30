from ultralytics import YOLO

model = YOLO("yolov8s.pt")

DATA_YAML = "data.yaml"
results = model.train(data=DATA_YAML, epochs=20, imgsz = 640)