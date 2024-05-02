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
yolov8_model_path = "models/yolov8s.pt"
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

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.01,
    device="cuda:0",  # or 'cuda:0'
)

# Load a model
# model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")

video_path = "data/match_video/out2.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()

cv.namedWindow("frame", cv.WINDOW_NORMAL)


y = 800
x = 700
h = 1500
w = 3000

cropped = frame[y:h, x:w]

# cv.imshow("frame", cropped)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cropped = copy(frame[y:h, x:w])

    # results = model(frame, show=True)  # predict on an image

    result = get_prediction(cropped, detection_model)

    show_img = copy(cropped)

    out = result.to_coco_predictions()
    for res in out:
        # breakpoint()

        res2 = [int(x) for x in res["bbox"]]
        if res["category_name"] != "person":
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
