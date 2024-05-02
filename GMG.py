import cv2 as cv


import numpy as np

from copy import copy


from ultralytics import YOLO


video_path = "data/match_video/out2.mp4"

video_path = "data/match_video/out2.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, 1050)


def foo(x):
    pass


cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.createTrackbar("dilate", "frame", 1, 10, foo)
cv.createTrackbar("opening", "frame", 1, 10, foo)
# cv.createTrackbar("alpha", "frame", 500, 1000, foo)
# cv.createTrackbar("kernel", "frame", 2, 5, foo)
# cv.createTrackbar("iter", "frame", 1, 10, foo)
# cv.createTrackbar("low_area", "frame", 200, 1000, foo)
cv.createTrackbar("high_area", "frame", 1400, 4000, foo)


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
    iter_o = cv.getTrackbarPos("opening", "frame")
    opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=iter_o)

    iter_d = cv.getTrackbarPos("dilate", "frame")
    kernel_d = np.ones((3, 3), np.uint8)
    dilate = cv.morphologyEx(opening, cv.MORPH_DILATE, kernel, iterations=iter_d)

    # find contours and filter by area
    contours, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    high_area = cv.getTrackbarPos("high_area", "frame")
    low_area = 0
    good = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < high_area:
            good.append(cnt)

    # draw contours
    area_mask = np.zeros_like(opening)

    area_mask = cv.drawContours(
        cv.cvtColor(area_mask, cv.COLOR_GRAY2BGR), good, -1, (0, 255, 0), -1
    )

    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    gray[area_mask[:, :, 1] == 0] = 0

    prep = copy(cropped)
    prep[area_mask[:, :, 1] == 0] = 0

    # output = model.predict(prep, show=True, classes=[0], device="cuda:0")

    show_img = copy(cropped)

    show_fgmask = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
    show_opening = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
    show_dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
    show_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.imshow("frame", np.vstack([show_img, show_opening, prep]))
    # breakpoint()
    if cv.waitKey(0) & 0xFF == ord("q"):
        break

    if cap.get(cv.CAP_PROP_POS_FRAMES) == 1250:
        cap.set(cv.CAP_PROP_POS_FRAMES, 1050)
# Use the model


cv.waitKey(0)
