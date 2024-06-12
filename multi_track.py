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


def detections_to_point(all_dets):
    if len(all_dets.keys()) < 2:
        return None

    trianglated_points = []
    checked = {}
    for cam_idx_1, det1 in all_dets.items():
        for cam_idx_2, det2 in all_dets.items():
            if cam_idx_1 == cam_idx_2:
                continue
            if (cam_idx_2, cam_idx_1) in checked:
                continue
            checked[(cam_idx_1, cam_idx_2)] = True
            cam1 = cams[cam_idx_1]
            cam2 = cams[cam_idx_2]

            point3d = cam1.triangulate(cam2, det1, det2)

            trianglated_points.append(point3d)

            # print(f"cam{cam_idx_1} -> cam{cam_idx_2} \t {point3d}")

    trianglated_points = np.array(trianglated_points)

    vec1 = torch.tensor(trianglated_points).unsqueeze(0)
    vec2 = torch.tensor(trianglated_points).unsqueeze(1)
    distances = torch.norm(vec1 - vec2, dim=2)

    good_points = []
    for i in range(distances.shape[0]):
        for j in range(i, distances.shape[1]):
            if distances[i][j] < 1000 and i != j:
                # print(f"{distances[i][j]} \t {trianglated_points[i]}")
                # print(f"{distances[i][j]}\t [{i} {j}]")
                if i not in good_points:
                    good_points.append(i)

    good_points = np.array([trianglated_points[i] for i in good_points])

    if len(good_points) == 0:
        return None

    final_point = np.mean(good_points, axis=0)
    return final_point


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

sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(640, 640), overlap=(0.1, 0.1))

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

    start = time.time()
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
    end = time.time() - start
    print(f"Time: {end}")
    frame_idx += 1

    final_point = detections_to_point(all_dets)

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
