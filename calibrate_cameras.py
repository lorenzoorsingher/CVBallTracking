import json
import cv2 as cv
import numpy as np

from setup import get_args_calib
from camera_controller import CameraController

args = get_args_calib()
if args["cameras"] == "-1":
    print("No camera selected, exiting...")
    exit()
cameras = {int(cam) for cam in args["cameras"].split(",")}


for camera_idx in cameras:

    print("[Calibration] Calibrating camera ", camera_idx)
    cam = CameraController(camera_idx)
    all_corners = cam.get_dump()

    all_points = [cam.get_chessboard() for _ in range(len(all_corners))]

    ret, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(
        all_points, all_corners, cam.imsize, None, None
    )
    print(f"[Calibration] Camera calibrated with {ret} error")
    cam.save_calib(camera_matrix, distortion_coefficients)
