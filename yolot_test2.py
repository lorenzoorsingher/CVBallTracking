import sahi.prediction
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv

from camera_controller import CameraController


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
import os

model_path = (
    "/home/lollo/Documents/python/CV/CVBallTracking/runs/detect/train2/weights/best.pt"
)
model = YOLO(model_path)

imgs_path = "/home/lollo/Documents/python/CV/CVBallTracking/yolotools/datasets/overfitter_mega_split/val/images"

img_paths = [f"{imgs_path}/{img}" for img in os.listdir(imgs_path)]

cv.namedWindow("frame", cv.WINDOW_NORMAL)
for img in img_paths:
    img = cv.imread(img)

    result = model.predict(img)
    print(len(result))

    boxes = result[0].boxes.xywh.cpu().tolist()

    for box in boxes:
        x, y, w, h = map(int, box)
        # breakpoint()
        cv.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2
        )
    cv.imshow("frame", img)
    cv.waitKey(0)
