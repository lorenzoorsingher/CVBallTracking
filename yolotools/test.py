import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')          ### Pre-trained weights

model = YOLO("runs/detect/train12/weights/best.pt")  ### weights from trained model
# model.conf = 0.4
# Open the video file
video_path = "data/video/out6.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
cv2.namedWindow("track", cv2.WINDOW_NORMAL)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    cropped = frame[700:1200, 0:1200]
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")
        results = model.predict(cropped, show=True, device="cuda:0")
        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # # Display the annotated frame
        # cv2.imshow("track", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
