import torch


# Model
# model = torch.hub.load(
#     "ultralytics/yolov5", "custom", path="data/weights/v5_best.pt", force_reload=True
# )  # or yolov5n - yolov5x6, custom

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
ckpt = torch.load("data/weights/v5_best.pt")
model.load_state_dict(ckpt["model"].state_dict())

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # , .save(), .crop(), .pandas(), etc.
