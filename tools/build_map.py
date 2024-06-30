import sys

sys.path.append(".")

import torch
import cv2 as cv
import json

from matplotlib import pyplot as plt
from ultralytics import YOLO

from sort import *
from common import set_axes_equal, get_postions

from camera_controller import CameraController
from yolotools.sliced_yolo import SlicedYOLO


def from_file():
    dump = []
    with open("data/detections_demo.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            frame_idx, cam_idx, x, y = line.strip().split(";")
            frame_idx = int(frame_idx)
            cam_idx = int(cam_idx)
            x = float(x)
            y = float(y)
            dump.append([frame_idx, cam_idx, x, y])

    steps = []
    new_step = {}
    frame_idx = -1
    while True:

        step = dump.pop(0)
        frm_idx, cam_idx, x, y = step
        if frame_idx != frm_idx:
            steps.append([frame_idx, new_step])
            frame_idx = frm_idx
            new_step = {}
        new_step[cam_idx] = [x, y]
        if len(dump) == 0:
            steps.append([frame_idx, new_step])
            steps.pop(0)
            break
    return steps


if __name__ == "__main__":

    positions, field_corners, _ = get_postions()

    # Scale the axes equally

    ################################

    cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
    videos_path = "data/fake_basket"
    video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
    cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
    caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]
    # trackers = [Sort() for _ in range(len(cam_idxs))]

    cv.namedWindow("frame", cv.WINDOW_NORMAL)

    curr_cam_idx = 4
    frame_skip = 10
    frame_idx = 200

    steps = from_file()

    plot_points = []
    final_point = None
    for step in steps[500:2000]:
        frm_idx, all_dets = step

        final_point = CameraController.detections_to_point(all_dets, cams, final_point)
        print(f"FINAL POINT: {final_point}")
        if final_point is not None:
            plot_points.append(final_point)

    from mpl_toolkits.mplot3d import Axes3D

    plot_points = np.array(plot_points).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(
        field_corners[:, 0],
        field_corners[:, 1],
        field_corners[:, 2],
        c="red",
        label="Real Corners",
    )
    set_axes_equal(ax)

    ax.plot(plot_points[0], plot_points[1], plot_points[2])
    ax.scatter(*plot_points.T[0], color="red")
    plt.show()
