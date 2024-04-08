import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
import json

x = -1
y = -1
img_corners = []
real_corners = []
curr_corner = 0


positions_path = "data/camera_data/camera_positions.json"
with open(positions_path, "r") as file:

    data = json.load(file)
    positions = data["positions"]
    field_corners = data["field_corners"]


def draw_field(img, cam_idx):
    offx = 400
    offy = 400

    field_corners2 = [[x * 50 for x in p] for p in field_corners]

    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (255, 255, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 255, 0),
        5,
    )

    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (255, 255, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 255, 0),
        5,
    )

    cv.line(
        img,
        (offx + field_corners2[3][0], offy + field_corners2[3][1]),
        (offx + field_corners2[6][0], offy + field_corners2[6][1]),
        (255, 255, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[2][0], offy + field_corners2[2][1]),
        (offx + field_corners2[7][0], offy + field_corners2[7][1]),
        (255, 255, 0),
        5,
    )
    cv.line(
        img,
        (offx + field_corners2[1][0], offy + field_corners2[1][1]),
        (offx + field_corners2[8][0], offy + field_corners2[8][1]),
        (255, 255, 0),
        5,
    )

    cam_pos = [x * 50 for x in positions[str(cam_idx)][0]]

    cv.circle(img, (offx + cam_pos[0], offy + cam_pos[1]), 25, (0, 255, 0), -1)

    # draw a point for every intersection, the first one it's red, while the others are yellow
    for i in range(10):
        if i == curr_corner:
            color = (0, 255, 255)
        else:
            color = (255, 0, 0)
        cv.circle(
            img,
            (offx + field_corners2[i][0], offy + field_corners2[i][1]),
            25,
            color,
            -1,
        )

    return img


def get_clicks(event, mouse_x, mouse_y, flags, param):
    global x
    global y

    if event == cv.EVENT_LBUTTONDOWN:
        x = mouse_x
        y = mouse_y


cam_idx = 1

cam = CameraController(cam_idx)

cap = cv.VideoCapture("data/video/out1F.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.setMouseCallback("frame", get_clicks)

loop = True

while loop:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    while True:
        for corner in img_corners:
            cv.circle(frame, corner, 15, (255, 0, 255), -1)

        copy_frame = cv.circle(copy(frame), (x, y), 15, (0, 0, 255), -1)

        copy_frame = draw_field(copy_frame, cam_idx)

        cv.imshow("frame", copy_frame)

        k = cv.waitKey(10)

        if k == ord("a"):
            img_corners.append((x, y))
            real_corners.append(field_corners[curr_corner])
            curr_corner += 1
            break
        if k == ord("s"):
            curr_corner += 1

        if k == ord("q"):
            loop = False
            break
        if k == ord("c"):
            break

        print("---------------------")
        print(curr_corner)
        print(real_corners)
        print(img_corners)

# corners2 = [[1000,2000],[1030,2450],[1077,2120],[2000,1300],[1800,1100]]
# corners = corners2
real_corners = np.array(real_corners, dtype=np.float32)
img_corners = np.array(img_corners, dtype=np.float32)


ret, rvecs, tvecs = cv.solvePnP(
    np.array(real_corners, dtype=np.float32),
    np.array(img_corners, dtype=np.float32),
    cam.mtx,
    cam.dist,
)


rotation_matrix, _ = cv.Rodrigues(rvecs)

inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
inv_tvecs = -np.dot(inverse_rotation_matrix, tvecs)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plotting real_corners
ax.scatter(
    real_corners[:, 0],
    real_corners[:, 1],
    real_corners[:, 2],
    c="blue",
    label="Real Corners",
)

# Plotting tvecs
ax.scatter(inv_tvecs[0][0], inv_tvecs[1][0], inv_tvecs[2][0], c="red", label="tvecs")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

plt.show()
