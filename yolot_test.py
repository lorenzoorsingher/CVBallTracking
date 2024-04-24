from ultralytics import YOLO
import cv2 as cv

# Load a model
model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")

video_path = "data/match_video/out2.mp4"

cap = cv.VideoCapture(video_path)

ret, frame = cap.read()

cv.namedWindow("frame", cv.WINDOW_NORMAL)


y = 800
x = 700
h = 1500
w = 3000

cropped = frame[y:h, x:w]

cv.imshow("frame", cropped)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cropped = frame[y:h, x:w]

    results = model(frame, show=True)  # predict on an image

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
# Use the model


cv.waitKey(0)
