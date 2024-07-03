from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# TRAINING
yaml_path = "yolotools/datasets/overfitter_og_split/data.yaml"


if __name__ == "__main__":
    results = model.train(data=yaml_path, epochs=50, patience=5)
