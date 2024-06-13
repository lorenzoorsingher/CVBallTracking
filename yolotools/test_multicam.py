import sys

sys.path.append(".")

import cv2 as cv
from ultralytics import YOLO

from camera_controller import CameraController
from yolotools.sliced_yolo import SlicedYOLO

model_path = "runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(640, 640), overlap=(0.05, 0.1))

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
videos_path = "data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]

cv.namedWindow("frame", cv.WINDOW_NORMAL)

curr_cam_idx = 4
frame_skip = 10
frame_idx = 1500
while True:

    cap = caps[curr_cam_idx]
    cam = cams[curr_cam_idx]

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        frame_idx = 0
        continue
    else:
        frame_idx += 1

    uframe = cam.undistort_img(frame)

    out, det, frame = sliced_yolo.predict(uframe, winz=True)

    if out is not None:

        x, y, w, h, c = out

        uframe = cv.rectangle(
            uframe, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 8
        )

    # frame = sliced_yolo.print_windows(uframe)

    cv.imshow("frame", frame)
    k = cv.waitKey(1)

    if k == ord("n"):
        curr_cam_idx = (curr_cam_idx + 1) % (len(cam_idxs))
        print(f"SWITCHED TO CAMERA {cam_idxs[curr_cam_idx]}")
    if k == ord("d"):
        print(f"SKIPPING {frame_skip} FRAMES")
        frame_idx += frame_skip
    if k == ord("q"):
        print("EXITING")
        break
