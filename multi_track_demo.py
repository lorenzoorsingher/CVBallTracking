import sys

sys.path.append(".")

import torch
import cv2 as cv
import json

from matplotlib import pyplot as plt
from ultralytics import YOLO

from sort import *
from common import set_axes_equal

from camera_controller import CameraController
from yolotools.sliced_yolo import SlicedYOLO
from build_map import from_file
from kalman import KalmanTracker


model_path = "weights/best.pt"

## FULL RESOLUTION
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(3840, 2160), overlap=(0, 0))
## MAX SPEED
sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(1920, 1130), overlap=(0.1, 0.1))
## BALANCED
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(1300, 1130), overlap=(0.05, 0.1))
## STANDARD
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(640, 640), overlap=(0.1, 0.1))

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
videos_path = "data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]


cv.namedWindow("frame", cv.WINDOW_NORMAL)


FRAME_SKIP = 10
START = 750
END = 1000


tracked_points = []
every_det = []

FROMFILE = False
if not FROMFILE:
    if START > 0:
        for cap in caps:
            cap.set(cv.CAP_PROP_POS_FRAMES, START)
else:
    steps = from_file()
    for idx, step in enumerate(steps):
        if step[0] >= START:
            break
    steps = steps[idx : idx + (END - START)]

frame_idx = START
final_point = None
while True:
    print(f"FRAME {frame_idx}-------------------------------")
    all_frames = []
    all_dets = {}
    ret = True

    if not FROMFILE:
        for curr_cam_idx in range(len(cam_idxs)):
            cap = caps[curr_cam_idx]
            cam = cams[curr_cam_idx]

            ret, frame = cap.read()
            if not ret:
                print("[TRACK] Frame corrupt, exiting...")
                exit()

            # undistorion used to be here, for triangulation
            # we need distorted images
            uframe = frame

            out, det, uframe = sliced_yolo.predict(uframe, viz=True)

            if out is not None:
                x, y, w, h, c = out
                all_dets[curr_cam_idx] = [x + w // 2, y + h // 2]
            all_frames.append(cv.resize(uframe, (640, 480)))
            frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    else:
        if len(steps) == 0:
            break
        step = steps.pop(0)
        frame_idx, all_dets = step

    final_point = CameraController.detections_to_point(all_dets, cams, final_point)

    if final_point is not None:
        tracked_points.append(final_point)

    if not FROMFILE:
        frame = np.vstack([np.hstack(all_frames[:4]), np.hstack(all_frames[4:])])
        if final_point is not None:
            cv.circle(frame, (50, 50), 30, (0, 255, 0), -1)
            cv.putText(
                frame,
                f"{len(all_dets)}",
                (40, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv.LINE_AA,
            )
        else:
            cv.circle(frame, (50, 50), 30, (0, 0, 255), -1)
        cv.imshow("frame", frame)
    k = cv.waitKey(1)
    if k == ord("d"):
        print(f"SKIPPING {FRAME_SKIP} FRAMES")
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx + FRAME_SKIP)
    if k == ord("q"):
        print("EXITING")
        break


###################### PLOTS #######################
tracked_points_np = np.array(tracked_points)
plot_points = np.array(tracked_points_np).T
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])
positions_path = "data/camera_data/camera_positions.json"
with open(positions_path, "r") as file:

    data = json.load(file)
    positions = data["positions"]
    field_corners = np.array(data["field_corners"]) * 1000
ax.scatter(
    field_corners[:, 0],
    field_corners[:, 1],
    field_corners[:, 2],
    c="red",
    label="Real Corners",
)
set_axes_equal(ax)
ax.plot(plot_points[0], plot_points[1], plot_points[2], color="blue")
ax.scatter(*plot_points.T[0], color="blue")
plt.show()
####################################################
