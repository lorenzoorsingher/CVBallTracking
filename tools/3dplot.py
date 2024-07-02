import sys

sys.path.append(".")

from matplotlib import pyplot as plt
import numpy as np
import json

from camera_controller import CameraController
from common import set_axes_equal, get_postions


cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]


positions, field_corners, real_locations = get_postions()

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

    real_pos = real_locations[str(cam_idx)]

    x = pos[0][0]
    y = pos[1][0]
    z = pos[2][0]

    r_x = real_pos[0]
    r_y = real_pos[1]
    r_z = real_pos[2]

    err_x = round(abs(r_x - x)) / 10
    err_y = round(abs(r_y - y)) / 10
    err_z = round(abs(r_z - z)) / 10
    dist = np.sqrt(err_x**2 + err_y**2 + err_z**2)
    perc = dist / np.sqrt(r_x**2 + r_y**2 + r_z**2) * 1000

    ax.scatter(r_x, r_y, r_z, c="pink", label="tvecs")
    ax.scatter(x, y, z, c="red", label="tvecs")
    ax.text(x, y, z, str(cam_idx))

    print(f"{cam_idx} -------------------------------------------------------")
    print(f"est pos:  \t{round(x)} \t{round(y)} \t{round(z)}")
    print(f"real pos: \t{r_x} \t{r_y} \t{r_z}")
    print(f"error:    \t{err_x}cm \t{err_y}cm \t{err_z}cm")
    print(f"distance: \t{round(dist,2)}cm\t {round(perc,2)}%")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


set_axes_equal(ax)
plt.show()
