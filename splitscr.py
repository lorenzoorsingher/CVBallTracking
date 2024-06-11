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
from sliced_yolo import SlicedYolo

model_path = (
    "/home/lollo/Documents/python/CV/CVBallTracking/runs/detect/train2/weights/best.pt"
)
model = YOLO(model_path)

sliced_yolo = SlicedYolo(model_path=model_path, wsize=(640, 720))

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
    sliced_yolo.predict(frame)
    if not ret:
        break
    fra_x = frame.shape[1]
    fra_y = frame.shape[0]

    n_y = fra_y // spl_size
    n_x = fra_x // spl_size

    windows = []
    for i in range(n_y):
        for j in range(n_x):
            x = j * spl_size
            y = i * spl_size
            # cv.rectangle(frame, (x, y), (x + spl_size, y + spl_size), (0, 255, 0), 2)
            windows.append(frame[y : y + spl_size, x : x + spl_size])
    # breakpoint()

    for window in windows:
        result = model.predict(window)
        boxes = result[0].boxes.xywh.cpu().tolist()
        for box in boxes:
            x, y, w, h = map(int, box)
            cv.rectangle(
                window,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 255, 0),
                2,
            )

    recomposed = np.zeros((fra_y, fra_x, 3), dtype=np.uint8)
    for i in range(n_y):
        for j in range(n_x):
            x = j * spl_size
            y = i * spl_size
            recomposed[y : y + spl_size, x : x + spl_size] = windows[i * n_x + j]

    cv.imshow("frame", recomposed)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
