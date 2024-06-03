import sahi.prediction
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image
import sahi

import numpy as np

from copy import copy

# Download YOLOv8 model
yolov8_model_path = "models/best.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)


model_path = "/home/lollo/Documents/python/yolo/runs/detect/train12/weights/best.pt"
model_path = "models/best_mogus.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.001,
    device="cuda:0",  # or 'cuda:0'
    image_size=640,
)

# Load a model
# model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")

video_path = "data/match_video/out2.mp4"

video_path = "/home/lollo/Documents/python/yolo/data/video/out6.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, 500)
cv.namedWindow("frame", cv.WINDOW_NORMAL)


y = 600
x = 0
h = 1500
w = 3000

cropped = frame[y:h, x:w]

# cv.imshow("frame", cropped)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # cropped = copy(frame[y:h, x:w])

    # results = model(frame, show=True)  # predict on an image

    SLICED = True

    if SLICED:
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
    else:
        result = get_prediction(frame, detection_model)

    show_img = copy(frame)

    out = result.to_coco_predictions()
    for res in out:
        # breakpoint()
        # print(res)
        res2 = [int(x) for x in res["bbox"]]
        if res["category_name"] == "ball":
            print(res["category_name"])
            x_det, y_det, w_det, h_det = res2
            cv.rectangle(
                show_img, (x_det, y_det), (x_det + w_det, y_det + h_det), (0, 255, 0), 2
            )

    cv.imshow("frame", show_img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
# Use the model


cv.waitKey(0)
