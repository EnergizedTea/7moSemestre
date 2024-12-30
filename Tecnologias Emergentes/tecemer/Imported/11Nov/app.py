from roboflow import Roboflow
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

model = YOLO("yolov8m.pt")
data_YAML = "Faces_V2-2/data.yaml"
results = model.train(data = data_YAML, epochs=100, save=True, verbose=True)
print(results)
