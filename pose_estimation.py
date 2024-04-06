import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
X = -1
Y = -1
corners = []
real_corners = np.array([[9000, 0, 0],
                        [9000, 6000, 0],
                        [9000, 9000, 0],
                        [0, 9000, 0],
                        [0, 6000, 0]], dtype=np.float32)


def get_clicks( event,mouse_x,mouse_y,flags,param):
        
        if event == cv.EVENT_LBUTTONDBLCLK:
            x = mouse_x
            y = mouse_y
            corners.append((x, y))

            print(corners)
        
cam = CameraController(3)

cap = cv.VideoCapture('data/video/out1F.mp4')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.setMouseCallback("frame", get_clicks)

loop = True

while loop:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    while cv.waitKey(10) != ord('q'):
        for corner in corners:
            cv.circle(frame, corner, 15, (0, 0, 255), -1)

        cv.imshow("frame", frame) 

        k = cv.waitKey(10) 
        if k == ord('q'):
            loop = False
            break
        if k == ord('c'):
            break

# corners2 = [[1000,2000],[1030,2450],[1077,2120],[2000,1300],[1800,1100]]
# corners = corners2

ret, rvecs, tvecs = cv.solvePnP(real_corners, np.array(corners, dtype=np.float32), cam.mtx, cam.dist)


rotation_matrix, _ = cv.Rodrigues(rvecs)

inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
inv_tvecs = -np.dot(inverse_rotation_matrix, tvecs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting real_corners
ax.scatter(real_corners[:, 0], real_corners[:, 1], real_corners[:, 2], c='blue', label='Real Corners')

# Plotting tvecs
ax.scatter(inv_tvecs[0][0], inv_tvecs[1][0], inv_tvecs[2][0], c='red', label='tvecs')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

plt.show()