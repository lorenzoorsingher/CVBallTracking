import sys

sys.path.append(".")

import os
import cv2 as cv
import numpy as np
from camera_controller import CameraController
from random import choice
import pickle
import pdb
from common import get_video_paths


def draw_corners(frame, corners, reprojected):

    for idx in range(len(corners)):
        cv.circle(frame, tuple(corners[idx][0].astype(int)), 5, (0, 0, 255), -1)
        cv.circle(frame, tuple(reprojected[idx][0].astype(int)), 5, (0, 255, 0), -1)

    return frame


def reprojection_error(corners, board, mtx, dist, frame):
    """
    Compute the reprojecton error of the current calibration parameters.
    First a corner detection is run on the image and the coordinates are saved
    in 'corners', later a projection of where the chessboard corners are supposed
    to be (according to extrinsic and intrinsic params) is run and the results
    are compared point by point. The resulting difference is calculated with
    root mean square deviation.
    Reliability describes how many corners have been detected and thus on how
    many data points over the maximum aviable have been used to calculate
    the error, it ranges from 0.0 to 1.0 where 1.0 means all corners have been
    detected
    """

    ret, rvecs, tvecs = cv.solvePnP(
        board,
        np.array(corners, dtype=np.float32),
        mtx,
        dist,
    )
    # run corner detection on image

    reprojected, r2 = cv.projectPoints(
        board,
        rvec=rvecs,
        tvec=tvecs,
        distCoeffs=dist,
        cameraMatrix=mtx,
    )

    sz = 20
    mul = 10
    sz *= mul
    target = np.zeros((sz, sz, 3), np.uint8)
    target.fill(255)
    cv.line(target, (0, sz // 2), (sz, sz // 2), (103, 157, 254), 1)
    cv.line(target, (sz // 2, 0), (sz // 2, sz), (103, 157, 254), 1)

    for i in range(len(corners)):

        error = (reprojected[i][0] - corners[i][0]) * mul

        cv.circle(
            target,
            (round(error[0] + sz // 2), round(error[1] + sz // 2)),
            1,
            (254, 90, 0),
            -1,
        )

    l2 = np.linalg.norm(corners - reprojected)

    if frame is not None:
        reprojection = draw_corners(frame, corners, reprojected)
    else:
        reprojection = None

    return target, reprojection, l2


video_paths = get_video_paths()
targets = []
old_targets = []
for camera_idx in [1, 2, 3, 4, 5, 6, 7, 8, 12]:

    cam = CameraController(camera_idx)
    chessboard_size = cam.chessboard_size

    pickle_file = (
        "../CV-CameraCalibration/output/pickle/parameters_out" + str(camera_idx) + "F.p"
    )
    dist_pickle = pickle.load(open(pickle_file, "rb"))
    old_mtx = dist_pickle["mtx"]
    old_dist = dist_pickle["dist"]

    cap = cv.VideoCapture(video_paths[camera_idx])

    vid_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    idxs = [x for x in range(0, vid_len)]

    cv.namedWindow("reprojection", cv.WINDOW_NORMAL)
    cv.namedWindow("target", cv.WINDOW_NORMAL)

    TEST_LEN = 5

    reprojections = []
    errors = []
    old_errors = []

    dump = cam.get_dump()
    board = cam.get_chessboard()

    for set in dump:
        found = False
        corners_refined = None

        target, rep, l2 = reprojection_error(set, board, cam.mtx, cam.dist, None)
        old_target, rep, old_l2 = reprojection_error(
            set, board, old_mtx, old_dist, None
        )

        errors.append(l2)
        old_errors.append(old_l2)

    targets.append(target)
    old_targets.append(old_target)
    # print("l2: ", l2)
    # print("old_l2: ", old_l2)

    print("\n\nCAMERA ", camera_idx)
    print("mean (ours) l2: \t", np.mean(errors))
    print("mean old_l2: \t\t", np.mean(old_errors))
    incr = 100 * (np.mean(old_errors) - np.mean(errors)) / abs(np.mean(errors))

    print(incr.round(2), " %")

cv.imshow("target", np.vstack([np.hstack(targets), np.hstack(old_targets)]))

# cv.imshow("reprojection", np.vstack(reprojections))
cv.waitKey(0)
