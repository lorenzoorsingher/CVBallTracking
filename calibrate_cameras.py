import json
import cv2 as cv
import numpy as np

from setup import get_args_calib
from camera_controller import CameraController

args = get_args_calib()
cameras = {int(cam) for cam in args["cameras"].split(",")}

sizes_path = "data/camera_data/chess_sizes.json"
with open(sizes_path, "r") as file:
    sizes = json.load(file)


for camera_idx in cameras:

    chessboard_size = sizes[str(camera_idx)]

    cam = CameraController(camera_idx)
    all_corners = cam.get_dump()

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )

    all_points = [objp for _ in range(len(all_corners))]

    ret, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(
        all_points, all_corners, cam.imsize, None, None
    )

    cam.save_calib(camera_matrix, distortion_coefficients)
    breakpoint()
