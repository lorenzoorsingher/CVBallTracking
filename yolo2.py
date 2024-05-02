import torch
import ultralytics

from ultralytics import YOLO

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv


import numpy as np

from copy import copy


weights = "models/v5_best.pt"

model = YOLO(weights)
model.to("cuda:0")
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
    output = model.predict(cropped, show=True, classes=[0], device="cuda:0")

    show_img = copy(cropped)

    # out = result.to_coco_predictions()
    # for res in out:
    #     # breakpoint()

    #     res2 = [int(x) for x in res["bbox"]]

    #     x_det, y_det, w_det, h_det = res2

    #     cv.rectangle(
    #         show_img, (x_det, y_det), (x_det + w_det, y_det + h_det), (0, 255, 0), 2
    #     )

    # cv.imshow("frame", show_img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
# Use the model


cv.waitKey(0)
