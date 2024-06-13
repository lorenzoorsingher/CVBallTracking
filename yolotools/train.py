from ultralytics import YOLO
import torch
import os

from ultralytics import YOLOv10

# Load the YOLOv8 model
model = YOLOv10.from_pretrained("jameslahm/yolov10s")

# TRAINING
yaml_path = "yolotools/datasets/overfitter_og_split/data.yaml"


if __name__ == "__main__":
    results = model.train(data=yaml_path, epochs=50, patience=5)
