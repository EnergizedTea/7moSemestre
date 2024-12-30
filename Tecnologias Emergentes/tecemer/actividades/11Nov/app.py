from roboflow import Roboflow
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
data_YAML = "Faces_V2-2/data.yaml"
results = model.train(data = data_YAML, epochs=10, save=True, verbose=True)
print(results)
