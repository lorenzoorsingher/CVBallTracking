import cv2 as cv
import numpy as np
from random import choice
from tqdm import tqdm

from setup import get_args_corners
from camera_controller import CameraController
from common import get_video_paths

from time import time

import random

# random.seed(0)
# np.random.seed(0)


def get_corners(image, chessboard_size):
    """
    Detects and refines the corners of a chessboard pattern in an image.

    Args:
        image (numpy.ndarray): The input image.
        chessboard_size (tuple): The size of the chessboard pattern (number of inner corners).

    Returns:
        tuple: A tuple containing the following:
            - ret (bool): True if the chessboard pattern is found, False otherwise.
            - corners_refined (numpy.ndarray): The refined corner coordinates if the pattern is found, None otherwise.
    """

    mididx = (chessboard_size[0] * chessboard_size[1]) // 2
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboard_size,
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
        + cv.CALIB_CB_NORMALIZE_IMAGE,
    )
    corners_refined = None
    midpoint = None
    if ret == True:

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(image, chessboard_size, corners_refined, ret)

        midpoint = corners_refined[mididx][0].astype(int)

    return ret, corners_refined, midpoint


def check_distance(point, midpoints, threshold):
    for mp in midpoints:
        dist = np.linalg.norm(mp - point)

        if dist < threshold:
            return True

    return False


def check_border(chk_corners, small_shape):

    min_x = chk_corners[:, 0, 0].min()
    max_x = chk_corners[:, 0, 0].max()
    min_y = chk_corners[:, 0, 1].min()
    max_y = chk_corners[:, 0, 1].max()

    THR = 30
    if min_x < THR or min_y < THR:
        return True

    if max_x > small_shape[0] - THR or max_y > small_shape[1] - THR:
        return True

    return False


def fast_detection(frame, chessboard_size, mid_points, canvas):
    # look for chessboard in lower resolution
    FAC = 4
    small_shape = (frame.shape[1] // FAC, frame.shape[0] // FAC)
    chk_frame = cv.resize(frame, small_shape)
    chk_ret, chk_corners, midpoint = get_corners(chk_frame, chessboard_size)

    got_corners = False
    tooclose = True
    corners_refined = None
    if chk_ret:
        midpoint = midpoint * FAC
        # check whether the point is too close to another one
        tooclose1 = check_distance(midpoint, mid_points, DISTANCE_THRESHOLD * 2)
        tooclose2 = check_border(chk_corners, small_shape)
        tooclose = tooclose1 or tooclose2
        if not tooclose:
            got_corners, corners_refined, _ = get_corners(frame, chessboard_size)

    return got_corners, tooclose, corners_refined, midpoint


args = get_args_corners()
if args["cameras"] == "-1":
    print("No camera selected, exiting...")
    exit()
cameras = {int(cam) for cam in args["cameras"].split(",")}
DETECT_NUM = args["detect_num"]
DISTANCE_THRESHOLD = args["distance_threshold"]

video_paths = get_video_paths()

cv.namedWindow("img", cv.WINDOW_NORMAL)

FAST_CHECK = True

for camera_idx in cameras:

    cam = CameraController(camera_idx)
    chessboard_size = cam.chessboard_size

    cap = cv.VideoCapture(video_paths[camera_idx])

    vid_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    idxs = [x for x in range(0, vid_len)]

    ret, frame = cap.read()

    canvas = np.full(frame.shape, 255, dtype=np.uint8)

    mid_points = []
    all_corners = []
    all_detections = []

    FRAME_PURGE = 5
    FRAME_PURGE_NF = 5

    pbar = tqdm(total=vid_len)
    # crono = time()
    while len(idxs) > 0 and len(all_corners) < DETECT_NUM:

        # print(round(time() - crono, 3))
        # crono = time()

        idx = choice(idxs)

        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            break

        ret, tooclose, corners, midpoint = fast_detection(
            frame, chessboard_size, mid_points, canvas
        )

        FP = FRAME_PURGE
        if ret:
            cv.circle(frame, midpoint, 10, (0, 255, 255), -1)
            all_corners.append(corners)
            mid_points.append(midpoint)
            cv.circle(canvas, midpoint, DISTANCE_THRESHOLD, (0, 255, 0), 5)
        elif tooclose:
            FP = FRAME_PURGE_NF
            cv.circle(canvas, midpoint, DISTANCE_THRESHOLD, (0, 0, 255), 2)

        to_be_removed = [x for x in range(idx - FP, idx + FP) if x in idxs]

        pbar.update(len(to_be_removed))
        pbar.set_description(
            f"cam {camera_idx} - {len(all_corners)}/{DETECT_NUM} detections [{len(all_detections)}]"
        )
        for x in to_be_removed:
            idxs.remove(x)

        cv.imshow("img", np.vstack([frame, canvas]))
        cv.waitKey(1)

    pbar.close()
    print(
        f"[calibration] Saving {len(all_corners)} corner sets for camera {camera_idx}..."
    )

    all_corners = np.array(all_corners)

    cam.save_dump(all_corners)
