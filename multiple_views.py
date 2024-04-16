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
cv.createTrackbar("scale", "frames", 8000, 10000, foo)
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

    frame = np.hstack([np.vstack(frames[:5]), np.vstack(frames[5:])])
    frame = cv.resize(frame, (1920, 2700))

    # dist_frame = np.hstack([np.vstack(dist_frames[:5]), np.vstack(dist_frames[5:])])
    # dist_frame = cv.resize(dist_frame, (1920, 2700))
    print("frame shape", frame.shape)
    X = 0
    Y = 0
    while True:

        copy_frame = cv.circle(copy(frame), (x, y), 10, (0, 0, 255), -1)

        # cv.imshow("dist_frames", dist_frame)

        k = cv.waitKey(10)

        if k == ord("q"):
            loop = False
            break
        if k == ord("c"):
            break

        # print("---------------------")
        # print("x ", x, " y", y)
        adj_x = x * divider
        adj_y = y * divider
        # print("x ", adj_x, " y", adj_y)

        y_rest = int(adj_y / 2160)
        x_rest = adj_x // 3840
        # print("y_rest", y_rest)
        # print("x_rest", x_rest)

        idx = y_rest + x_rest * 5
        # print("cam: ", cams[idx].index)

        cam_y = adj_y % 2160
        cam_x = adj_x - x_rest * 3840

        for num, camid in enumerate(cam_idxs):
            cam_tmp = CameraController(camid)
            # objp = np.array([X, Y, 0], np.float32)

            dmp = np.array(cam_tmp.get_img_corners()[0]) * 1000

            imgp = cam_tmp.repoject_points(dmp)

            for imp in imgp:
                uimp = cv.undistortPoints(imp, cam_tmp.mtx, cam_tmp.dist, P=cam_tmp.mtx)
                xp, yp = uimp[0][0]
                x_og = 3840 * (num // 5)
                y_og = 2160 * (num % 5)

                if xp < 0 or yp < 0:
                    continue
                if xp > 3840 or yp > 2160:
                    continue

                x_adj = x_og // divider + int(xp // divider)
                y_adj = y_og // divider + int(yp // divider)
                cv.circle(copy_frame, (x_adj, y_adj), 5, (0, 255, 0), -1)

            xr, yr = cam_tmp.repoject_points(np.array([X, Y, 0], np.float32))[0][0]
            # print("xr", xr, "yr", yr)
            xp, yp = cv.undistortPoints(
                np.array([xr, yr], dtype=np.float32),
                cam_tmp.mtx,
                cam_tmp.dist,
                P=cam_tmp.mtx,
            )[0][0]
            x_og = 3840 * (num // 5)
            y_og = 2160 * (num % 5)

            if xp < 0 or yp < 0:
                continue
            if xp > 3840 or yp > 2160:
                continue

            x_adj = x_og // divider + int(xp // divider)
            y_adj = y_og // divider + int(yp // divider)
            cv.circle(copy_frame, (x_adj, y_adj), 10, (0, 255, 255), -1)
            # print("imgp", (xp, yp))
        # print("cam_y ", cam_y)
        # print("cam_x ", cam_x)

        # hom_vec = np.array([cam_x, cam_y, 1])

        # inv_mtx = np.linalg.inv(cams[idx].mtx)

        # rvecs, tvecs = cams[idx].get_camera_position()

        # cam_vec = np.dot(inv_mtx, hom_vec)
        # hom_vec = np.array([0, 0, 0, 1], dtype=np.float32)
        # hom_vec[:3] = cam_vec
        # rot_mtx, _ = cv.Rodrigues(rvecs)

        # T = np.eye(4, 4)
        # T[:3, :3] = rot_mtx
        # T[:3, 3] = tvecs.T
        # # breakpoint()
        # world_vec = np.dot(T, hom_vec)
        # # world_vec = world_vec[:3] / np.linalg.norm(world_vec)
        # # rotation_matrix, _ = cv.Rodrigues(self.rvecs)

        # p0 = Point3D(tvecs[0][0], tvecs[1][0], tvecs[2][0])  # point in line

        # # create plane and line

        # line = Line3D(p0, direction_ratio=world_vec[:3])

        # plane = Plane(Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 0))
        # # line = Line3D([x[0] for x in tvecs], direction_ratio=world_vec[:3])

        # # # find intersection:

        # intr = plane.intersection(line)

        # print("world_vec", world_vec)
        # print("camera  pos", [x[0] for x in tvecs])
        # print(
        #     f"intersection: {round(intr[0].x.evalf(),2)} {round(intr[0].y.evalf(),2)} {round(intr[0].z.evalf(),2)}"
        # )

        # cv::Matx33f R_cam_chessboard;
        # cv::Rodrigues(rvec_cam_chessboard, R_cam_chessboard);
        # cv::Matx33f R_chessboard_cam = R_cam_chessboard.t();
        # cv::Matx31f t_cam_chessboard = tvec_cam_chessboard;
        # cv::Matx31f pos_cam_wrt_chessboard = -R_chessboard_cam*t_cam_chessboard;
        # // Map the ray direction vector from camera coordinates to chessboard coordinates
        # cv::Matx31f ray_dir_chessboard = R_chessboard_cam * ray_dir_cam;

        # ray_dir_cam = np.array([cam_x, cam_y, 1])
        # R_cam_chessboard = cv.Rodrigues(cam.rvecs)[0]
        # R_chessboard_cam = R_cam_chessboard.T
        # t_cam_chessboard = cam.tvecs
        # pos_cam_wrt_chessboard = -R_chessboard_cam * t_cam_chessboard
        # ray_dir_chessboard = R_chessboard_cam * ray_dir_cam
        # # breakpoint()

        # # float d_intersection = -pos_cam_wrt_chessboard.val[2]/ray_dir_chessboard.val[2];
        # # cv::Matx31f intersection_point = pos_cam_wrt_chessboard + d_intersection * ray_dir_chessboard;

        # d_intersection = -pos_cam_wrt_chessboard[2] / ray_dir_chessboard[2]
        # intersection_point = (
        #     pos_cam_wrt_chessboard + d_intersection * ray_dir_chessboard
        # )
        # # print(f"X: {round(X,1)} Y: {round(Y,1)}")

        #############################################################

        cam_idx = cam_idxs[idx]
        print("cur_cam ", cam_idx)
        cur_cam = CameraController(cam_idx)

        rotm, _ = cv.Rodrigues(cur_cam.rvecs)
        tvec = np.array([x[0] for x in cur_cam.tvecs])

        scale = cv.getTrackbarPos("scale", "frames")
        ux = (cam_x - 3840 / 2) / scale
        vx = (cam_y - 2160 / 2) / scale

        print("cam: ", ux, " ", vx)
        Tx, Ty, Tz = rotm.T @ -tvec
        # dx, dy, dz = rotm.T @ np.array([(cam_x - 3840 / 2), cam_y - 2160 / 2, 1])
        dv = rotm.T @ np.array([ux, vx, 1])
        dv /= np.linalg.norm(dv)
        dx, dy, dz = dv

        X = (-Tz / dz) * dx + Tx
        Y = (-Tz / dz) * dy + Ty
        # breakpoint()
        # breakpoint()
        print(f"X: {round(X,1)} Y: {round(Y,1)}")
        print(f"d: {dx, dy, dz}")

        # cam_idx = cam_idxs[idx]
        # print("cur_cam ", cam_idx)
        # cur_cam = CameraController(cam_idx)
        # uvPoint = np.array([(cam_x - 3840 / 2), cam_y - 2160 / 2, 1])
        # rotationMatrix = cv.Rodrigues(cur_cam.rvecs)[0]
        # cameraMatrix = cur_cam.mtx
        # tvec = np.array([x[0] for x in cur_cam.tvecs])

        # leftSideMat = rotationMatrix.T @ np.linalg.inv(cameraMatrix) @ uvPoint
        # rightSideMat = rotationMatrix.T @ tvec
        # sf = rightSideMat[2] / leftSideMat[2]

        # P = np.linalg.inv(rotationMatrix) @ (
        #     sf * np.linalg.inv(cameraMatrix) @ uvPoint - tvec
        # )
        # X, Y, Z = P
        # print(uvPoint)
        # print(f"X: {round(X,1)} Y: {round(Y,1)} Z: {round(Z,1)}")
        cv.imshow("frames", copy_frame)
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
