import cv2 as cv
import numpy as np
from random import choice
from tqdm import tqdm

from setup import get_args_corners
from camera_controller import CameraController
from common import get_video_paths


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

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(image, chessboard_size, corners_refined, ret)

    return ret, corners_refined


args = get_args_corners()
if args["cameras"] == "-1":
    print("No camera selected, exiting...")
    exit()
cameras = {int(cam) for cam in args["cameras"].split(",")}
DETECT_NUM = args["detect_num"]
DISTANCE_THRESHOLD = args["distance_threshold"]

video_paths = get_video_paths()

cv.namedWindow("img", cv.WINDOW_NORMAL)


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

    FRAME_PURGE = 5
    FRAME_PURGE_NF = 5

    pbar = tqdm(total=vid_len)
    while len(idxs) > 0 and len(all_corners) < DETECT_NUM:
        idx = choice(idxs)

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

                if dist < DISTANCE_THRESHOLD * 2:
                    tooclose = True
                    break

            if not tooclose:
                mid_points.append(midpoint)
                cv.circle(canvas, midpoint, DISTANCE_THRESHOLD, (0, 255, 0), 10)

                all_corners.append(corners_refined)
            else:
                cv.circle(canvas, midpoint, DISTANCE_THRESHOLD, (0, 0, 255), 2)
            cv.circle(frame, midpoint, 10, (0, 255, 255), -1)
            FP = FRAME_PURGE
        else:
            FP = FRAME_PURGE_NF

        to_be_removed = [x for x in range(idx - FP, idx + FP) if x in idxs]

        pbar.update(len(to_be_removed))
        pbar.set_description(
            f"cam {camera_idx} - {len(all_corners)}/{DETECT_NUM} detections"
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
