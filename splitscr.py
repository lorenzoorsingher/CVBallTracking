import os
import sys

sys.path.append(".")

import json
import numpy as np
import cv2 as cv
import shutil
from camera_controller import CameraController
from common import get_video_paths
import random
import time

from ultralytics import YOLO
from yolotools.sliced_yolo import SlicedYolo

model_path = (
    "/home/lollo/Documents/python/CV/CVBallTracking/runs/detect/train2/weights/best.pt"
)
model = YOLO(model_path)

sliced_yolo = SlicedYolo(model_path=model_path, wsize=(640, 640), overlap=0.1)

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
ann_num = [0, 0, 0, 0, 0, 0, 0, 0]
videos_path = "/home/lollo/Documents/python/CV/CVBallTracking/data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]
idx = 4
cap = caps[idx]
cam = cams[idx]
cv.namedWindow("frame", cv.WINDOW_NORMAL)
cap.set(cv.CAP_PROP_POS_FRAMES, 1500)


spl_size = 640

while True:
    ret, frame = cap.read()

    frame = cam.undistort_img(frame)

    if not ret:
        break

    out, det = sliced_yolo.predict(frame)

    if out is not None:

        x, y, w, h, c = out

        frame = cv.rectangle(
            frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 8
        )

    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
