from matplotlib import pyplot as plt
import numpy as np
import json

import cv2 as cv
from camera_controller import CameraController
from common import set_axes_equal


videos_path = "data/fake_basket/"
cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
cam_idxs = [8, 7, 4, 2]

caps = [
    cv.VideoCapture(videos_path + "out" + str(cam_idx) + ".mp4") for cam_idx in cam_idxs
]

fgbgs = [cv.createBackgroundSubtractorMOG2() for _ in cam_idxs]

for cap in caps:
    cap.set(cv.CAP_PROP_POS_FRAMES, 500)

cv.namedWindow("stack", cv.WINDOW_NORMAL)
cv.namedWindow("stack_bg", cv.WINDOW_NORMAL)
while True:

    frames = [cap.read()[1] for cap in caps]

    for idx, cam_idx in enumerate(cam_idxs):

        cam = CameraController(cam_idx)

        frames[idx] = cam.undistort_img(frames[idx])

    frames = [cv.resize(frame, (960, 540)) for frame in frames]

    lr = 0.001
    fgmasks = [fgbg.apply(frame, lr) for frame, fgbg in zip(frames, fgbgs)]

    thresholded = [
        cv.threshold(fgmask, 100, 255, cv.THRESH_BINARY)[1] for fgmask in fgmasks
    ]

    opened = [
        cv.morphologyEx(thre, cv.MORPH_OPEN, (5, 5), iterations=2)
        for thre in thresholded
    ]

    dilated = [cv.dilate(open, (5, 5), iterations=3) for open in opened]

    fgmasks = dilated
    stack_bg = np.hstack(
        [np.vstack(fgmasks[: len(frames) // 2]), np.vstack(fgmasks[len(frames) // 2 :])]
    )

    masks = dilated
    masked = [
        cv.bitwise_and(frame, frame, mask=mask) for frame, mask in zip(frames, masks)
    ]
    frames = masked
    stack = np.hstack(
        [np.vstack(frames[: len(frames) // 2]), np.vstack(frames[len(frames) // 2 :])]
    )

    cv.imshow("stack", stack)
    cv.imshow("stack_bg", stack_bg)
    cv.waitKey(1)
