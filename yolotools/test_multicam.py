import sys

sys.path.append(".")

import cv2 as cv
from ultralytics import YOLO

from camera_controller import CameraController
from yolotools.sliced_yolo import SlicedYOLO

model_path = "weights/best.pt"
model = YOLO(model_path)

## FULL RESOLUTION
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(3840, 2160), overlap=(0, 0))
## MAX SPEED
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(1920, 1130), overlap=(0.1, 0.1))
## BALANCED
# sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(1300, 1130), overlap=(0.05, 0.1))
## STANDARD
sliced_yolo = SlicedYOLO(model_path=model_path, wsize=(640, 640), overlap=(0.1, 0.1))

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
videos_path = "data/fake_basket"
video_paths = [f"{videos_path}/out{cam_idx}.mp4" for cam_idx in cam_idxs]
cams = [CameraController(cam_idx) for cam_idx in cam_idxs]
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]

cv.namedWindow("frame", cv.WINDOW_NORMAL)

print("Press 'n' to switch camera")
print("Press 'd' to skip 10 frames")
print("Press 'a' to go back 10 frames")
print("Press 'q' to exit")

curr_cam_idx = 0
FRAME_SKIP = 10
START = 750
for cap in caps:
    cap.set(cv.CAP_PROP_POS_FRAMES, START)
frame_idx = START
while True:

    cap = caps[curr_cam_idx]
    cam = cams[curr_cam_idx]

    ret, frame = cap.read()
    frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    if not ret:
        print("[TRACK] Frame corrupt, exiting...")
        exit()

    out, det, frame = sliced_yolo.predict(frame, winz=True)

    if out is not None:
        x, y, w, h, c = out
        frame = cv.rectangle(
            frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 8
        )

    cv.imshow("frame", frame)
    k = cv.waitKey(1)

    if k == ord("n"):
        curr_cam_idx = (curr_cam_idx + 1) % (len(cam_idxs))
        print(f"SWITCHED TO CAMERA {cam_idxs[curr_cam_idx]}")
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    if k == ord("d"):
        print(f"SKIPPING {FRAME_SKIP} FRAMES")
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx + FRAME_SKIP)
        frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    if k == ord("a"):
        print(f"GOING BACK {FRAME_SKIP} FRAMES")
        if frame_idx - FRAME_SKIP > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx - FRAME_SKIP)
            frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    if k == ord("q"):
        print("EXITING")
        break
