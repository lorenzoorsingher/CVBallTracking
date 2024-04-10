import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
import json
import os

from setup import get_args_pose
from common import set_axes_equal

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


cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]


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
