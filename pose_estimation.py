import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
import json
import os

from setup import get_args_pose
from common import set_axes_equal, get_postions, get_video_paths

args = get_args_pose()
if args["camera"] == -1:
    print("No camera selected, exiting...")
    exit()
CAM_IDX = args["camera"]
REUSE = args["reuse"]


x = -1
y = -1


positions, field_corners, _ = get_postions()

field_corners = field_corners // 1000


def draw_field(img, cam_idx, curr_corner):
    offx = 600
    offy = 400
    multiplier = 25
    field_corners2 = [
        [int(x * multiplier), int(y * -multiplier), z] for x, y, z in field_corners
    ]

    # breakpoint()
    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (255, 0, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 0, 0),
        5,
    )

    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (255, 0, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 0, 0),
        5,
    )

    cv.line(
        img,
        (offx + field_corners2[3][0], offy + field_corners2[3][1]),
        (offx + field_corners2[6][0], offy + field_corners2[6][1]),
        (255, 0, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[2][0], offy + field_corners2[2][1]),
        (offx + field_corners2[7][0], offy + field_corners2[7][1]),
        (255, 0, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[1][0], offy + field_corners2[1][1]),
        (offx + field_corners2[8][0], offy + field_corners2[8][1]),
        (255, 0, 0),
        5,
    )

    cv.line(
        img,
        (offx + field_corners2[10][0], offy + field_corners2[10][1]),
        (offx + field_corners2[14][0], offy + field_corners2[14][1]),
        (255, 255, 255),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[17][0], offy + field_corners2[17][1]),
        (offx + field_corners2[21][0], offy + field_corners2[21][1]),
        (255, 255, 255),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[17][0], offy + field_corners2[17][1]),
        (offx + field_corners2[14][0], offy + field_corners2[14][1]),
        (255, 255, 255),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[10][0], offy + field_corners2[10][1]),
        (offx + field_corners2[21][0], offy + field_corners2[21][1]),
        (255, 255, 255),
        5,
    )
    cam_pos = positions[str(cam_idx)][0]
    rcam_pos = [int(cam_pos[0] * multiplier), int(-cam_pos[1] * multiplier)]
    cv.circle(img, (offx + rcam_pos[0], offy + rcam_pos[1]), 25, (0, 255, 0), -1)

    # draw a point for every intersection, the first one it's red, while the others are yellow
    for i in range(len(field_corners2)):
        if i == curr_corner:
            color = (0, 255, 255)
        else:
            color = (255, 0, 0)
        cv.circle(
            img,
            (offx + field_corners2[i][0], offy + field_corners2[i][1]),
            15,
            color,
            -1,
        )

    return img


def get_clicks(event, mouse_x, mouse_y, flags, param):
    """
    Callback function for mouse events.

    Parameters:
    - event: The type of mouse event.
    - mouse_x: The x-coordinate of the mouse click.
    - mouse_y: The y-coordinate of the mouse click.
    - flags: Additional flags for the mouse event.
    - param: Additional parameters passed to the callback function.

    Returns:
    None
    """
    global x
    global y

    if event == cv.EVENT_LBUTTONDOWN:
        x = mouse_x
        y = mouse_y


cam = CameraController(CAM_IDX)


def get_real_points(cam_idx):
    """
    Get the real-world coordinates and corresponding image coordinates of points clicked on the video frame.

    Args:
        cam_idx (int): The index of the camera/video source.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: A tuple containing two lists:
            - real_corners: The real-world coordinates of the clicked points.
            - img_corners: The corresponding image coordinates of the clicked points.
    """
    # Function code here
    pass


def get_real_points(cam_idx):

    img_corners = []
    real_corners = []
    curr_corner = 0

    video_paths = get_video_paths()

    cap = cv.VideoCapture(video_paths[cam_idx])

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.setMouseCallback("frame", get_clicks)

    print("Press 'a' to Add a point")
    print("Press 's' to Skip a point")
    print("Press 'r' to Remove the last point")
    print("Press 'c' to Continue with the video")
    print("Press 'q' to Quit or continue")
    loop = True
    while loop:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        while True:

            copy_frame = cv.circle(copy(frame), (x, y), 4, (0, 0, 255), -1)

            for corner in img_corners:
                cv.circle(copy_frame, corner, 4, (255, 0, 255), -1)

            copy_frame = draw_field(copy_frame, cam_idx, curr_corner)

            cv.imshow("frame", copy_frame)

            k = cv.waitKey(10)

            if k == ord("a"):
                img_corners.append((x, y))
                real_corners.append(field_corners[curr_corner])
                curr_corner += 1
            if k == ord("s"):
                curr_corner += 1

            if k == ord("q"):
                loop = False
                break
            if k == ord("r"):
                img_corners.pop()
                real_corners.pop()
            if k == ord("c"):
                break

            print("---------------------")
            print(curr_corner)
            print(real_corners)
            print(img_corners)

    return real_corners, img_corners


if REUSE:
    real_corners, img_corners = cam.get_img_corners()
else:
    real_corners, img_corners = get_real_points(CAM_IDX)
    cam.save_img_corners(real_corners, img_corners)

real_corners = np.array(real_corners, dtype=np.float32) * 1000
img_corners = np.array(img_corners, dtype=np.float32)


_, inv_tvecs = cam.estimate_camera_position(real_corners, img_corners)


# Scale the axes equally
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])

# Plotting real_corners
ax.scatter(
    real_corners[:, 0],
    real_corners[:, 1],
    real_corners[:, 2],
    c="blue",
    label="Real Corners",
)

ax.set_box_aspect([1, 1, 1])

# Plotting tvecs
ax.scatter(inv_tvecs[0][0], inv_tvecs[1][0], inv_tvecs[2][0], c="red", label="tvecs")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

set_axes_equal(ax)
plt.show()
