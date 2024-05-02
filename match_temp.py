import cv2 as cv


import numpy as np

from copy import copy


weights = "models/v5_best.pt"

video_path = "data/match_video/out2.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, 1050)


def foo(x):
    pass


cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.createTrackbar("kernel_g", "frame", 8, 10, foo)
cv.createTrackbar("alpha", "frame", 500, 1000, foo)
cv.createTrackbar("kernel", "frame", 2, 5, foo)
cv.createTrackbar("iter", "frame", 1, 10, foo)
cv.createTrackbar("low_area", "frame", 200, 1000, foo)
cv.createTrackbar("high_area", "frame", 1400, 4000, foo)


y = 200
x = 700
h = 1800
w = 3000

cropped = frame[y:h, x:w]

balls = [cv.imread(f"data/balls/ball_{i}.png") for i in range(1, 5)]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cropped = copy(frame[y:h, x:w])

    show_img = copy(cropped)

    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

    res = []
    for ball in balls:
        gray_ball = cv.cvtColor(ball, cv.COLOR_BGR2GRAY)
        out = cv.matchTemplate(gray, gray_ball, cv.TM_CCOEFF_NORMED)

        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(out, None)
        res.append((_maxVal, maxLoc))

    for maxVal, maxLoc in res:
        cv.rectangle(
            show_img,
            maxLoc,
            (maxLoc[0] + ball.shape[1], maxLoc[1] + ball.shape[0]),
            (0, 255, 0),
            2,
        )
    cv.imshow("frame", show_img)
    # breakpoint()
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    if cap.get(cv.CAP_PROP_POS_FRAMES) == 1250:
        cap.set(cv.CAP_PROP_POS_FRAMES, 1050)
# Use the model


cv.waitKey(0)
