import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy

from common import get_video_paths

x = -1
y = -1


def mouse_to_img(x, y, div):
    """
    Converts mouse coordinates to image coordinates.

    Args:
        x (int): The x-coordinate of the mouse.
        y (int): The y-coordinate of the mouse.
        div (int): The division factor.

    Returns:
        tuple: A tuple containing the index, camera x-coordinate, and camera y-coordinate.
    """
    adj_x = x * div
    adj_y = y * div

    y_rest = int(adj_y / 2160)
    x_rest = adj_x // 3840

    idx = y_rest + x_rest * 5

    cam_y = adj_y % 2160
    cam_x = adj_x - x_rest * 3840

    return idx, cam_x, cam_y


def img_to_mouse(idx, x, y, div):
    """
    Converts image coordinates to mouse coordinates based on the given index, x, y, and division factor.

    Parameters:
    - idx (int): The index of the image.
    - x (int): The x-coordinate in the image.
    - y (int): The y-coordinate in the image.
    - div (int): The division factor used for scaling.

    Returns:
    tuple: The converted x and y coordinates in the mouse coordinate system.
    """
    x_og = 3840 * (idx // 5)
    y_og = 2160 * (idx % 5)

    x_adj = x_og // div + int(x // div)
    y_adj = y_og // div + int(y // div)

    return x_adj, y_adj

#flags e param non li usiamo mai, servono?
def get_clicks(event, mouse_x, mouse_y, flags, param):
    """
    Callback function to handle mouse clicks.

    Parameters:
    - event: The type of mouse event.
    - mouse_x: The x-coordinate of the mouse click.
    - mouse_y: The y-coordinate of the mouse click.
    - flags: Additional flags for the mouse event.
    - param: Additional parameters passed to the callback function.
    """

    global x
    global y

    if event == cv.EVENT_LBUTTONDOWN:
        x = mouse_x
        y = mouse_y


cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]


video_paths = get_video_paths()


cv.namedWindow("frames", cv.WINDOW_NORMAL)


cv.setMouseCallback("frames", get_clicks)

cams = [CameraController(cam_idx) for cam_idx in cam_idxs]

caps = [cv.VideoCapture(video_paths[cam_idx]) for cam_idx in cam_idxs]

loop = True

divider = 4

while loop:

    dist_frames = [cap.read()[1] for cap in caps]

    frames = []
    for idx, frame in enumerate(dist_frames):
        frames.append(cv.undistort(frame, cams[idx].mtx, cams[idx].dist))

    frame = np.hstack([np.vstack(frames[:5]), np.vstack(frames[5:])])
    frame = cv.resize(frame, (1920, 2700))

    print("frame shape", frame.shape)
    X = 0
    Y = 0
    while True:

        copy_frame = cv.circle(copy(frame), (x, y), 10, (0, 0, 255), -1)

        k = cv.waitKey(10)

        if k == ord("q"):
            loop = False
            break
        if k == ord("c"):
            break
        idx, cam_x, cam_y = mouse_to_img(x, y, divider)

        for num, camid in enumerate(cam_idxs):
            cam_tmp = CameraController(camid)

            dmp = np.array(cam_tmp.get_img_corners()[0]) * 1000
            imgp, _ = cv.projectPoints(
                dmp, cam_tmp.rvecs, cam_tmp.tvecs, cam_tmp.mtx, None
            )

            for imp in imgp:
                uimp = imp
                xp, yp = uimp[0]  # [0][0]
                x_adj, y_adj = img_to_mouse(num, xp, yp, divider)
                cv.circle(copy_frame, (x_adj, y_adj), 5, (0, 255, 0), -1)

            point, _ = cv.projectPoints(
                np.array([[X, Y, 0]], dtype=np.float32),
                cam_tmp.rvecs,
                cam_tmp.tvecs,
                cam_tmp.mtx,
                None,
            )
            xr, yr = point[0][0]

            xp, yp = xr, yr

            if xp < 0 or yp < 0:
                continue
            if xp > 3840 or yp > 2160:
                continue

            x_adj, y_adj = img_to_mouse(num, xp, yp, divider)
            cv.circle(copy_frame, (x_adj, y_adj), 10, (0, 255, 255), -1)

        cam_idx = cam_idxs[idx]
        cur_cam = CameraController(cam_idx)

        rotm, _ = cv.Rodrigues(cur_cam.rvecs)
        tvec = np.array([x[0] for x in cur_cam.tvecs])

        fx = cur_cam.mtx[0][0]
        fy = cur_cam.mtx[1][1]
        cx = cur_cam.mtx[0][2]
        cy = cur_cam.mtx[1][2]

        unx, uny = cam_x, cam_y
        ux = (unx - cx) / fx
        vx = (uny - cy) / fy

        Tx, Ty, Tz = rotm.T @ -tvec
        dv = rotm.T @ np.array([ux, vx, 1])
        dx, dy, dz = dv

        X = (-Tz / dz) * dx + Tx
        Y = (-Tz / dz) * dy + Ty

        cv.imshow("frames", copy_frame)
