import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
import json
import os

from sympy import Plane, Line3D, Rational

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


cv.namedWindow("frames", cv.WINDOW_NORMAL)
cv.namedWindow("dist_frames", cv.WINDOW_NORMAL)

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
    while True:

        copy_frame = cv.circle(copy(frame), (x, y), 10, (0, 0, 255), -1)

        cv.imshow("frames", copy_frame)
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
        # print("cam_y ", cam_y)
        # print("cam_x ", cam_x)

        hom_vec = np.array([cam_x, cam_y, 1])

        inv_mtx = np.linalg.inv(cams[idx].mtx)

        cam_vec = np.dot(inv_mtx, hom_vec)
        hom_vec = np.array([0, 0, 0, 1], dtype=np.float32)
        hom_vec[:3] = cam_vec
        rot_mtx, _ = cv.Rodrigues(cams[idx].rvecs)
        tvecs = cams[idx].tvecs
        T = np.eye(4, 4)
        T[:3, :3] = rot_mtx
        T[:3, 3] = tvecs.T
        # breakpoint()
        world_vec = np.dot(T, hom_vec)
        world_vec = world_vec[:3] / np.linalg.norm(world_vec)
        # rotation_matrix, _ = cv.Rodrigues(self.rvecs)

        plane = Plane([1, 0, 0], [3, 2, 0], [2, 2, 0])
        line = Line3D([x[0] for x in tvecs], direction_ratio=world_vec)

        # find intersection:

        intr = plane.intersection(line)

        print(f"{hom_vec} ")
        print("world_vec", world_vec)

        print(
            f"intersection: {round(intr[0].x.evalf(),2)} {round(intr[0].y.evalf(),2)} {round(intr[0].z.evalf(),2)}"
        )
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
positions_path = "data/camera_data/camera_positions.json"
with open(positions_path, "r") as file:

    data = json.load(file)
    positions = data["positions"]
    field_corners = np.array(data["field_corners"])


# Scale the axes equally
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])

# Plotting real_corners
ax.scatter(
    field_corners[:, 0],
    field_corners[:, 1],
    field_corners[:, 2],
    c="blue",
    label="Real Corners",
)

for cam_idx in cam_idxs:

    cam = CameraController(cam_idx)

    rot, pos = cam.get_camera_position()
    ax.scatter(pos[0][0], pos[1][0], pos[2][0], c="red", label="tvecs")
    ax.text(pos[0][0], pos[1][0], pos[2][0], str(cam_idx))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


set_axes_equal(ax)
plt.show()
