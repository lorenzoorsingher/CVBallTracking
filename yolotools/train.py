from ultralytics import YOLO
import torch
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# TRAINING
yaml_path = "/home/lollo/Documents/python/yolo/datasets/mogus/data.yaml"

if __name__ == "__main__":
    results = model.train(data=yaml_path, epochs=50, patience=5)
