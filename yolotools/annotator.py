import os
import sys

sys.path.append(".")

import json
import numpy as np
import cv2 as cv
import shutil
from camera_controller import CameraController
from common import get_video_paths


videos_path = "/home/lollo/Documents/python/CV/CVBallTracking/data/fake_basket"
cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]

video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]

cams = [CameraController(cam_idx) for cam_idx in cam_idxs]

caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]

cv.namedWindow("frames", cv.WINDOW_NORMAL)

curr_cam_idx = 0
frame_skip = 10
while True:

    cap = caps[curr_cam_idx]
    ret, frame = cap.read()
    if not ret:
        curr_cam_idx = (curr_cam_idx + 1) % (len(cam_idxs))
        break

    cv.imshow("frames", frame)
    k = cv.waitKey(0)
    if k == ord("a"):
        print("SAVING ANNOTATION")
    if k == ord("s"):
        print("SKIPPING FRAME")
    if k == ord("d"):
        print(f"SKIPPING {frame_skip} FRAMES")
        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + frame_skip)
    if k == ord("n"):
        curr_cam_idx = (curr_cam_idx + 1) % (len(cam_idxs))
        print(f"NEXT CAMERA {cam_idxs[curr_cam_idx]}")
    if k == ord("q"):
        print("EXITING")
        break
