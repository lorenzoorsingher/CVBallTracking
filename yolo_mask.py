import cv2 as cv


import numpy as np

from copy import copy


from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict


# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.01,
    device="cuda:0",  # or 'cuda:0'
)

video_path = "data/match_video/out2.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, 1050)


def foo(x):
    pass


cv.namedWindow("frame", cv.WINDOW_NORMAL)
# cv.createTrackbar("kernel_g", "frame", 8, 10, foo)
# cv.createTrackbar("alpha", "frame", 500, 1000, foo)
# cv.createTrackbar("kernel", "frame", 2, 5, foo)
# cv.createTrackbar("iter", "frame", 1, 10, foo)
# cv.createTrackbar("low_area", "frame", 200, 1000, foo)
# cv.createTrackbar("high_area", "frame", 1400, 4000, foo)


y = 200
x = 700
h = 1800
w = 3000

cropped = frame[y:h, x:w]

# cv.imshow("frame", cropped)

fgbg = cv.createBackgroundSubtractorMOG2()

balls = [cv.imread(f"data/balls/ball_{i}.png") for i in range(1, 5)]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cropped = copy(frame[y:h, x:w])

    show_img = copy(cropped)

    # background subtraction
    lr = 0.5
    fgmask = fgbg.apply(cropped, lr)
    fgmask[fgmask < 200] = 0

    # morphological operations
    ksize = 5
    kernel = np.ones((ksize, ksize), np.uint8)
    opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=1)

    kernel_d = np.ones((3, 3), np.uint8)
    dilate = cv.morphologyEx(opening, cv.MORPH_DILATE, kernel, iterations=3)

    # find contours and filter by area
    contours, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    low_area = 350
    good = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > low_area:
            good.append(cnt)

    # draw contours
    area_mask = np.zeros_like(opening)

    area_mask = cv.drawContours(
        cv.cvtColor(area_mask, cv.COLOR_GRAY2BGR), good, -1, (0, 255, 0), -1
    )

    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    gray[area_mask[:, :, 1] == 0] = 0

    yolo_mask = copy(gray)
    result = get_prediction(cropped, detection_model)
    out = result.to_coco_predictions()
    for res in out:
        # breakpoint()

        res2 = [int(x) for x in res["bbox"]]
        x_det, y_det, w_det, h_det = res2

        cv.rectangle(
            gray, (x_det, y_det), (x_det + w_det, y_det + h_det), (0, 0, 0), -1
        )

    show_img = copy(cropped)

    show_fgmask = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
    show_opening = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
    show_dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
    show_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.imshow("frame", np.vstack([show_img, show_gray]))
    # breakpoint()
    if cv.waitKey(0) & 0xFF == ord("q"):
        break

    if cap.get(cv.CAP_PROP_POS_FRAMES) == 1250:
        cap.set(cv.CAP_PROP_POS_FRAMES, 1050)
# Use the model


cv.waitKey(0)
