import sys

sys.path.append(".")

import random
import time
import os

import cv2 as cv

from camera_controller import CameraController
from common import get_video_paths


drawing = False  # true if mouse is pressed
start, end = (-1, -1), (-1, -1)


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global drawing, start, end

    if event == cv.EVENT_LBUTTONDOWN:

        if drawing == False:
            drawing = True
            start = (x, y)
            end = (-1, -1)
        else:
            drawing = False
            end = (x, y)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            end = (x, y)


videos_path = "/home/lollo/Documents/python/CV/CVBallTracking/data/fake_basket"

timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
dataset_path = f"/home/lollo/Documents/python/CV/CVBallTracking/yolotools/datasets/fakebasket_{timestamp}"


imgs_path = f"{dataset_path}/images"
labels_path = f"{dataset_path}/labels"

os.mkdir(dataset_path)
os.mkdir(imgs_path)
os.mkdir(labels_path)

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
ann_num = [0, 0, 0, 0, 0, 0, 0, 0]

video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]

cams = [CameraController(cam_idx) for cam_idx in cam_idxs]

caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]

cv.namedWindow("frames", cv.WINDOW_NORMAL)
cv.setMouseCallback("frames", draw_circle)


curr_cam_idx = 0
frame_skip = 10
while True:

    cap = caps[curr_cam_idx]
    cam = cams[curr_cam_idx]
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    uframe = cam.undistort_img(frame)

    while True:
        dframe = uframe.copy()

        if start != (-1, -1) and end != (-1, -1):
            cv.rectangle(dframe, start, end, (0, 255, 0), 3)

        cv.imshow("frames", dframe)
        k = cv.waitKey(1)
        if k != -1:
            break

    if k == ord("f"):
        print("SAVING ANNOTATION")

        if start == (-1, -1) or end == (-1, -1):
            print("NO ANNOTATION TO SAVE")
            continue
        print(f"START: {start} END: {end}")
        startx, starty = start
        endx, endy = end
        n_startx = startx / uframe.shape[1]
        n_starty = starty / uframe.shape[0]
        n_endx = endx / uframe.shape[1]
        n_endy = endy / uframe.shape[0]

        center = ((n_startx + n_endx) / 2, (n_starty + n_endy) / 2)
        width = abs(n_startx - n_endx)
        height = abs(n_starty - n_endy)

        filename = (
            f"cam_{cam_idxs[curr_cam_idx]}_ann_{ann_num[curr_cam_idx]}_{timestamp}"
        )

        cv.imwrite(f"{imgs_path}/{filename}.jpg", uframe)
        with open(f"{labels_path}/{filename}.txt", "w") as f:
            f.write(f"0 {center[0]} {center[1]} {width} {height}")

        start, end = (-1, -1), (-1, -1)
        drawing = False
        ann_num[curr_cam_idx] += 1

        str_ann = ""
        for idx, ann in enumerate(ann_num):
            str_ann += f"cam_{idx}->{ann}\t"
        print(str_ann)
    if k == ord("s"):
        print("SKIPPING FRAME")
    if k == ord("d"):
        print(f"SKIPPING {frame_skip} FRAMES")
        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + frame_skip)
    if k == ord("a"):
        print(f"MOVING BACK {frame_skip} FRAMES")
        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) - frame_skip)
    if k == ord("r"):
        print(f"JUMP TO RANDOM FRAME")
        rand_frame = random.randint(0, cap.get(cv.CAP_PROP_FRAME_COUNT) - 1)
        cap.set(cv.CAP_PROP_POS_FRAMES, rand_frame)
    if k == ord("n"):
        curr_cam_idx = (curr_cam_idx + 1) % (len(cam_idxs))
        print(f"SWITCHED TO CAMERA{cam_idxs[curr_cam_idx]}")
    if k == ord("q"):
        print("EXITING")
        break
