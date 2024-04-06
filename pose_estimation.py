import numpy as np
import cv2 as cv
from camera_controller import CameraController
from matplotlib import pyplot as plt
from copy import copy
x = -1
y = -1
img_corners = []
real_corners = []
click_count = 0
# real_corners = np.array([[9000, 0, 0],
#                         [9000, 6000, 0],
#                         [9000, 9000, 0],
#                         [0, 9000, 0],
#                         [0, 6000, 0]], dtype=np.float32)

field_corners = np.array([[0,0,0],
                            [0,6,0],
                            [0,9,0],
                            [0,12,0],
                            [0,18,0],
                            [9,18,0],
                            [9,12,0],
                            [9,9,0],
                            [9,6,0],
                            [9,0,0],
                          ], dtype=np.float32) * 50

def draw_field(img):
    offx = 300
    offy = 300

    field_corners2 = field_corners.copy().astype(int)

    cv.line(img, (offx + field_corners2[0][0], offy+field_corners2[0][1]), (offx + field_corners2[4][0], offy+field_corners2[4][1]), (255, 255, 0), 5)
    cv.line(img, (offx + field_corners2[9][0], offy+field_corners2[9][1]), (offx + field_corners2[5][0], offy+field_corners2[5][1]), (255, 255, 0), 5)

    cv.line(img, (offx + field_corners2[0][0], offy+field_corners2[0][1]), (offx + field_corners2[9][0], offy+field_corners2[9][1]), (255, 255, 0), 5)
    cv.line(img, (offx + field_corners2[4][0], offy+field_corners2[4][1]), (offx + field_corners2[5][0], offy+field_corners2[5][1]), (255, 255, 0), 5)

    cv.line(img, (offx + field_corners2[3][0], offy+field_corners2[3][1]), (offx + field_corners2[6][0], offy+field_corners2[6][1]), (255, 255, 0), 5)
    cv.line(img, (offx + field_corners2[2][0], offy+field_corners2[2][1]), (offx + field_corners2[7][0], offy+field_corners2[7][1]), (255, 255, 0), 5)
    cv.line(img, (offx + field_corners2[1][0], offy+field_corners2[1][1]), (offx + field_corners2[8][0], offy+field_corners2[8][1]), (255, 255, 0), 5)

#draw a point for every intersection, the first one it's red, while the others are yellow
    for i in range(10):
        if i == click_count:
            color = (0, 255,255) 
        else:
            color = (255, 0, 0)
        cv.circle(img, (offx + field_corners2[i][0], offy + field_corners2[i][1]), 25, color, -1)

    return img

def get_clicks( event,mouse_x,mouse_y,flags,param):
        global x
        global y

        if event == cv.EVENT_LBUTTONDBLCLK:
            x = mouse_x
            y = mouse_y
            # click_count += 1
            # corners.append((x, y))

            # print(corners)
        
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
        for corner in img_corners:
            cv.circle(frame, corner, 15, (255, 0, 255), -1)

        copy_frame = cv.circle(copy(frame), (x,y), 15, (0, 0, 255), -1)

        copy_frame = draw_field(copy_frame)

        cv.imshow("frame", copy_frame) 

        k = cv.waitKey(10) 

        if k == ord('a'):
            img_corners.append((x, y))
            real_corners.append(field_corners[click_count])
            click_count += 1
            break
        if k == ord('s'):
            click_count += 1

        if k == ord('q'):
            loop = False
            break
        if k == ord('c'):
            break
        
        print("---------------------")
        print(click_count)
        print(real_corners)
        print(img_corners)

# corners2 = [[1000,2000],[1030,2450],[1077,2120],[2000,1300],[1800,1100]]
# corners = corners2

# ret, rvecs, tvecs = cv.solvePnP(real_corners, np.array(corners, dtype=np.float32), cam.mtx, cam.dist)


# rotation_matrix, _ = cv.Rodrigues(rvecs)

# inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
# inv_tvecs = -np.dot(inverse_rotation_matrix, tvecs)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plotting real_corners
# ax.scatter(real_corners[:, 0], real_corners[:, 1], real_corners[:, 2], c='blue', label='Real Corners')

# # Plotting tvecs
# ax.scatter(inv_tvecs[0][0], inv_tvecs[1][0], inv_tvecs[2][0], c='red', label='tvecs')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.legend()

# plt.show()