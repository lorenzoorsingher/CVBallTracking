import os
import json

import cv2 as cv
import numpy as np
from random import randint, choice

from calib_setup import get_args


def get_corners(image, chessboard_size):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboard_size,
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
        + cv.CALIB_CB_NORMALIZE_IMAGE,
    )
    corners_refined = None
    if ret == True:
        # objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # imgpoints.append(corners_refined)

        # Draw and display the corners
        cv.drawChessboardCorners(image, chessboard_size, corners_refined, ret)

    # print("[corner detection] finished.")
    return ret, corners_refined


args = get_args()
cameras = {int(cam) for cam in args["cameras"].split(",")}

video_dir = "data/video/"
video_paths = {int(name[3]): video_dir + name for name in os.listdir(video_dir)}

sizes_path = "data/camera_data/chess_sizes.json"
with open(sizes_path, "r") as file:
    sizes = json.load(file)


cv.namedWindow("img", cv.WINDOW_NORMAL)


for video_idx in cameras:

    chessboard_size = sizes[str(video_idx)]
    cap = cv.VideoCapture(video_paths[video_idx])

    vid_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    idxs = [x for x in range(0, vid_len)]
    # cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + 25)

    ret, frame = cap.read()
    canvas = np.full(frame.shape, 255, dtype=np.uint8)
    mid_points = []

    MINDIST = 150
    FRAME_PURGE = 5
    FRAME_PURGE_NF = 5
    while len(idxs) > 0:

        idx = choice(idxs)

        print("len: ", len(idxs))
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            break

        got_corners, corners_refined = get_corners(frame, chessboard_size)
        tooclose = False

        if got_corners:
            mididx = (chessboard_size[0] * chessboard_size[1]) // 2
            midpoint = corners_refined[mididx][0].astype(int)

            for mp in mid_points:
                dist = np.linalg.norm(mp - midpoint)

                if dist < MINDIST * 2:
                    tooclose = True
                    break

            if not tooclose:
                mid_points.append(midpoint)
                cv.circle(canvas, midpoint, MINDIST, (0, 255, 0), 5)
            else:
                cv.circle(canvas, midpoint, MINDIST, (0, 0, 255), 2)
            cv.circle(frame, midpoint, 10, (0, 255, 255), -1)
            FP = FRAME_PURGE
        else:
            FP = FRAME_PURGE_NF

        to_be_removed = [x for x in range(idx - FP, idx + FP) if x in idxs]
        for x in to_be_removed:
            idxs.remove(x)
        cv.imshow("img", np.vstack([frame, canvas]))
        cv.waitKey(1)
    cv.waitKey(0)
