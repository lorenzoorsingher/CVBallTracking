import sys

sys.path.append(".")

import torch
import cv2 as cv
import numpy as np
import json

from matplotlib import pyplot as plt
from ultralytics import YOLO

from common import set_axes_equal, get_postions

from camera_controller import CameraController
from yolotools.sliced_yolo import SlicedYOLO
from setup import get_args_demo
from tools.build_map import from_file
from tracker import Tracker

args = get_args_demo()

START = args["start"]
END = args["end"]
FROMFILE = args["from_file"]
MODE = args["mode"]
FRAME_SKIP = 10
PRINTPATH = False

weight = "weights/best.pt"

if MODE == 1:
    print("[SlicedYOLO] FULL RESOLUTION")
    sliced_yolo = SlicedYOLO(model_path=weight, wsize=(3840, 2160), overlap=(0, 0))
elif MODE == 2:
    print("[SlicedYOLO] MAX SPEED")
    sliced_yolo = SlicedYOLO(model_path=weight, wsize=(1920, 1130), overlap=(0.1, 0.1))
elif MODE == 3:
    print("[SlicedYOLO] BALANCED")
    sliced_yolo = SlicedYOLO(model_path=weight, wsize=(1300, 1130), overlap=(0.05, 0.1))
elif MODE == 4:
    print("[SlicedYOLO] STANDARD")
    sliced_yolo = SlicedYOLO(model_path=weight, wsize=(640, 640), overlap=(0.1, 0.1))


cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
videos_path = "data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]
trackers = [Tracker(idx) for idx in cam_idxs]

cv.namedWindow("frame", cv.WINDOW_NORMAL)

tracked_points = []
every_det = []

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


detecs = {}

for idx in cam_idxs:
    detecs[idx] = []

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

            track_x = -1
            track_y = -1
            if out is not None:
                x, y, w, h, c = out
                # TODO: fix detection center
                all_dets[curr_cam_idx] = [x, y]

                detecs[cam_idxs[curr_cam_idx]].append([x, y, w, h])

            all_frames.append(cv.resize(uframe, (640, 360)))
            frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    else:
        if len(steps) == 0:
            break
        step = steps.pop(0)
        frame_idx, all_dets = step

    filt_dets = {}
    for idx, tracker in enumerate(trackers):

        if cam_idxs[idx] not in all_dets:
            point = tracker.update(None)
        else:
            point = tracker.update(all_dets[cam_idxs[idx]])
        if point is not None:
            filt_dets[cam_idxs[idx]] = point

    all_dets = filt_dets

    final_point = CameraController.detections_to_point(all_dets, cams, final_point)

    if final_point is not None:
        tracked_points.append(final_point)

    if not FROMFILE:

        if PRINTPATH:
            if len(tracked_points) > 0:
                for curr_cam_idx in range(len(cam_idxs)):
                    cam = cams[curr_cam_idx]
                    projections = cam.project_points(tracked_points)
                    prev = None
                    curr = None
                    for pp in projections[:, 0]:
                        # breakpoint()
                        point = pp // 6
                        curr = tuple(point.astype(np.int16))
                        if prev is None:
                            prev = curr
                        cv.line(all_frames[curr_cam_idx], prev, curr, (255, 255, 0), 2)
                        prev = curr

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

        for cap in caps:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx + FRAME_SKIP)
    if k == ord("q"):
        print("EXITING")
        break


with open("detcs.txt", "w") as f:

    for idx in cam_idxs:
        for det in detecs[idx]:
            f.write(f"{idx} {' '.join([str(x) for x in det])}\n")
###################### PLOTS #######################
tracked_points_np = np.array(tracked_points)
plot_points = np.array(tracked_points_np).T
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])

positions, field_corners, _ = get_postions()

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
