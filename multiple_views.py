import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
import json
import os

from sympy import Plane, Line3D, Rational, Point3D

from setup import get_args_pose
from common import set_axes_equal
from common import get_video_paths

x = -1
y = -1


def mouse_to_img(x, y, div):
    adj_x = x * div
    adj_y = y * div

    y_rest = int(adj_y / 2160)
    x_rest = adj_x // 3840

    idx = y_rest + x_rest * 5

    cam_y = adj_y % 2160
    cam_x = adj_x - x_rest * 3840

    return idx, cam_x, cam_y


def img_to_mouse(idx, x, y, div):
    x_og = 3840 * (idx // 5)
    y_og = 2160 * (idx % 5)

    x_adj = x_og // div + int(x // div)
    y_adj = y_og // div + int(y // div)

    return x_adj, y_adj


def get_clicks(event, mouse_x, mouse_y, flags, param):
    global x
    global y

    if event == cv.EVENT_LBUTTONDOWN:
        x = mouse_x
        y = mouse_y


cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]


video_paths = get_video_paths()


def foo():
    pass


cv.namedWindow("frames", cv.WINDOW_NORMAL)
cv.createTrackbar("scalex", "frames", 500, 1000, foo)
cv.createTrackbar("scaley", "frames", 500, 1000, foo)

# cv.namedWindow("dist_frames", cv.WINDOW_NORMAL)

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
        # frames.append(frame)

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
            # objp = np.array([X, Y, 0], np.float32)

            dmp = np.array(cam_tmp.get_img_corners()[0]) * 1000
            # imgp = cam_tmp.repoject_points(dmp)
            imgp, _ = cv.projectPoints(
                dmp, cam_tmp.rvecs, cam_tmp.tvecs, cam_tmp.mtx, None
            )

            for imp in imgp:
                # uimp = cv.undistortPoints(imp, cam_tmp.mtx, cam_tmp.dist, P=cam_tmp.mtx)
                uimp = imp
                xp, yp = uimp[0]  # [0][0]
                x_adj, y_adj = img_to_mouse(num, xp, yp, divider)
                cv.circle(copy_frame, (x_adj, y_adj), 5, (0, 255, 0), -1)

            # xr, yr = cam_tmp.repoject_points(np.array([X, Y, 0], np.float32))[0][0]
            point, _ = cv.projectPoints(
                np.array([[X, Y, 0]], dtype=np.float32),
                cam_tmp.rvecs,
                cam_tmp.tvecs,
                cam_tmp.mtx,
                None,
            )
            xr, yr = point[0][0]
            # breakpoint()
            # xp, yp = cv.undistortPoints(
            #     np.array([xr, yr], dtype=np.float32),
            #     cam_tmp.mtx,
            #     cam_tmp.dist,
            #     P=cam_tmp.mtx,
            # )[0][0]
            xp, yp = xr, yr

            if xp < 0 or yp < 0:
                continue
            if xp > 3840 or yp > 2160:
                continue

            x_adj, y_adj = img_to_mouse(num, xp, yp, divider)
            cv.circle(copy_frame, (x_adj, y_adj), 10, (0, 255, 255), -1)

        cam_idx = cam_idxs[idx]
        print("cur_cam ", cam_idx)
        cur_cam = CameraController(cam_idx)

        rotm, _ = cv.Rodrigues(cur_cam.rvecs)
        tvec = np.array([x[0] for x in cur_cam.tvecs])

        scalex = cv.getTrackbarPos("scalex", "frames")
        scaley = cv.getTrackbarPos("scaley", "frames")
        # breakpoint()

        # unx, uny = cv.undistortPoints(
        #     np.array([[cam_x, cam_y]], dtype=np.float32),
        #     cur_cam.mtx,
        #     cur_cam.dist,
        #     P=cur_cam.mtx,
        # )[0][0]

        fx = cur_cam.mtx[0][0]
        fy = cur_cam.mtx[1][1]
        cx = cur_cam.mtx[0][2]
        cy = cur_cam.mtx[1][2]

        unx, uny = cam_x, cam_y
        ux = (unx - cx) / fx
        vx = (uny - cy) / fy

        # ux = unx / cur_cam.mtx[0][0] - cur_cam.mtx[0][2] / 4000
        # vx = uny / cur_cam.mtx[1][1] - cur_cam.mtx[1][2] / 4000

        print("cam: ", ux, " ", vx)
        Tx, Ty, Tz = rotm.T @ -tvec
        # dx, dy, dz = rotm.T @ np.array([(cam_x - 3840 / 2), cam_y - 2160 / 2, 1])
        dv = rotm.T @ np.array([ux, vx, 1])
        # dv /= np.linalg.norm(dv)
        dx, dy, dz = dv

        X = (-Tz / dz) * dx + Tx
        Y = (-Tz / dz) * dy + Ty

        print(f"X: {round(X,1)} Y: {round(Y,1)}")
        # print(f"d: {dx, dy, dz}")

        cv.imshow("frames", copy_frame)
