import json
import cv2 as cv
import numpy as np

from setup import get_args_calib
from camera_controller import CameraController

args = get_args_calib()
cameras = {int(cam) for cam in args["cameras"].split(",")}


for camera_idx in cameras:

    cam = CameraController(camera_idx)
    all_corners = cam.get_dump()

    all_points = [cam.get_chessboard() for _ in range(len(all_corners))]

    ret, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(
        all_points, all_corners, cam.imsize, None, None
    )

    cam.save_calib(camera_matrix, distortion_coefficients)