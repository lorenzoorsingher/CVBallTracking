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


positions_path = "data/camera_data/camera_positions.json"
with open(positions_path, "r") as file:

    data = json.load(file)
    positions = data["positions"]
    field_corners = np.array(data["field_corners"]) * 1000
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])
plt.ion()
plt.show()
ax.scatter(
    field_corners[:, 0],
    field_corners[:, 1],
    field_corners[:, 2],
    c="red",
    label="Real Corners",
)
set_axes_equal(ax)


def plot_3d_points(points):

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="blue",
        label="Real Corners",
    )
    plt.pause(1)


model_path = (
    "/home/lollo/Documents/python/CV/CVBallTracking/runs/detect/train2/weights/best.pt"
)
model = YOLO(model_path)

sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(640, 640), overlap=(0.8, 0.8))

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
videos_path = "/home/lollo/Documents/python/CV/CVBallTracking/data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]
# trackers = [Sort() for _ in range(len(cam_idxs))]


cv.namedWindow("frame", cv.WINDOW_NORMAL)

frame_skip = 10
frame_idx = 200


tracked_points = []
every_det = []
while True:
    print(f"FRAME {frame_idx}-------------------------------")
    all_frames = []
    all_dets = {}
    ret = True
    for curr_cam_idx in range(len(cam_idxs)):
        cap = caps[curr_cam_idx]
        cam = cams[curr_cam_idx]
        # curr_tracker = trackers[curr_cam_idx]

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("[TRACK] Frame corrupted, exiting...")
            exit()

        # uframe = cam.undistort_img(frame)
        uframe = frame

        out, det, uframe = sliced_yolo.predict(uframe, viz=True)

        if out is not None:
            x, y, w, h, c = out
            all_dets[curr_cam_idx] = [x + w // 2, y + h // 2]
        all_frames.append(cv.resize(uframe, (640, 480)))

    frame_idx += 1

    final_point = CameraController.detections_to_point(all_dets, cams)

    if final_point is None:
        continue

    print(f"FINAL POINT: {final_point}")

    # tracked_points.append(final_point)
    # tracked_points_np = np.array(tracked_points)
    # plot_3d_points(tracked_points_np)

    frame = np.vstack([np.hstack(all_frames[:4]), np.hstack(all_frames[4:])])
    cv.imshow("frame", frame)
    k = cv.waitKey(1)
    if k == ord("d"):
        print(f"SKIPPING {frame_skip} FRAMES")
        frame_idx += frame_skip
    if k == ord("q"):
        print("EXITING")
        break

plt.pause(100000000)
