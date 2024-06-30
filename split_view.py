import cv2 as cv
import numpy as np

from camera_controller import CameraController
from common import get_video_paths, get_postions

cam_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

positions, field_corners, _ = get_postions()
field_corners = (field_corners // 1000).tolist()

x = -1
y = -1


def draw_field(img, cam_idx1, cam_idx2):
    offx = 600
    offy = 400
    multiplier = 25
    field_corners2 = [
        [int(x * multiplier), int(y * -multiplier), z] for x, y, z in field_corners
    ]

    # breakpoint()
    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (255, 0, 0),
        10,
    )
    cv.line(
        img,
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 0, 0),
        10,
    )

    cv.line(
        img,
        (offx + field_corners2[0][0], offy + field_corners2[0][1]),
        (offx + field_corners2[9][0], offy + field_corners2[9][1]),
        (255, 0, 0),
        10,
    )
    cv.line(
        img,
        (offx + field_corners2[4][0], offy + field_corners2[4][1]),
        (offx + field_corners2[5][0], offy + field_corners2[5][1]),
        (255, 0, 0),
        10,
    )

    cv.line(
        img,
        (offx + field_corners2[3][0], offy + field_corners2[3][1]),
        (offx + field_corners2[6][0], offy + field_corners2[6][1]),
        (255, 0, 0),
        10,
    )
    cv.line(
        img,
        (offx + field_corners2[2][0], offy + field_corners2[2][1]),
        (offx + field_corners2[7][0], offy + field_corners2[7][1]),
        (255, 0, 0),
        10,
    )
    cv.line(
        img,
        (offx + field_corners2[1][0], offy + field_corners2[1][1]),
        (offx + field_corners2[8][0], offy + field_corners2[8][1]),
        (255, 0, 0),
        10,
    )

    for idxs in cam_idxs:
        cam_pos = positions[str(idxs)][0]
        rcam_pos = [int(cam_pos[0] * multiplier), int(-cam_pos[1] * multiplier)]

        if idxs == cam_idx1:
            color = (0, 255, 0)
        elif idxs == cam_idx2:
            color = (255, 0, 255)
        else:
            color = (255, 255, 255)

        cv.circle(
            img,
            (offx + int(rcam_pos[0] // 1.5), offy + int(rcam_pos[1] // 1.5)),
            50,
            color,
            -1,
        )
        cv.addText(
            img,
            f"{idxs}",
            (offx + int(rcam_pos[0] // 1.5) - 25, offy + int(rcam_pos[1] // 1.5) + 20),
            "ArialBold",
            40,
        )

    return img


def mouse_to_img(x, y):

    if x > 3840:
        return -1, -1, -1
    if y <= 2160:
        return 0, x, y
    else:
        return 1, x, y - 2160


# flags e param non li usiamo mai, servono?
def get_clicks(event, mouse_x, mouse_y, flags, param):
    """
    Callback function to handle mouse clicks.

    Parameters:
    - event: The type of mouse event.
    - mouse_x: The x-coordinate of the mouse click.
    - mouse_y: The y-coordinate of the mouse click.
    - flags: Additional flags for the mouse event.
    - param: Additional parameters passed to the callback function.
    """

    global x
    global y

    if event == cv.EVENT_LBUTTONDOWN:
        x = mouse_x
        y = mouse_y


cv.namedWindow("frames", cv.WINDOW_NORMAL)

cv.setMouseCallback("frames", get_clicks)

video_paths = get_video_paths()


cams = [CameraController(cam_idx) for cam_idx in cam_idxs]

caps = [cv.VideoCapture(video_paths[cam_idx]) for cam_idx in cam_idxs]

frees = [1, 1000, 2125, 1100, 1425, 6, 7, 250, 12, 13]

og_frames = [cap.read()[1] for cap in caps]
og_frames = []

for cap, cam, free in zip(caps, cams, frees):
    cap.set(cv.CAP_PROP_POS_FRAMES, free)
    ret, frame = cap.read()
    uframe = cv.undistort(frame, cam.mtx, cam.dist)
    og_frames.append(uframe)


print("press 'a' to cycle through the cameras on the top quadrant (green)")
print("press 'd' to cycle through the cameras on the bottom quadrant (purple)")
print("press 'q' to exit")

camidxs = [0, 1]

X = 0
Y = 0
while True:

    frames = [frame.copy() for frame in og_frames]
    camid1 = cam_idxs[camidxs[0]]
    camid2 = cam_idxs[camidxs[1]]

    id, cam_x, cam_y = mouse_to_img(x, y)
    camid = cam_idxs[camidxs[id]]

    cur_cam = cams[camidxs[id]]

    rotm, _ = cv.Rodrigues(cur_cam.rvecs)
    tvec = np.array([x[0] for x in cur_cam.tvecs])
    fx = cur_cam.mtx[0][0]
    fy = cur_cam.mtx[1][1]
    cx = cur_cam.mtx[0][2]
    cy = cur_cam.mtx[1][2]

    unx, uny = cam_x, cam_y
    ux = (unx - cx) / fx
    vx = (uny - cy) / fy

    Tx, Ty, Tz = rotm.T @ -tvec
    dv = rotm.T @ np.array([ux, vx, 1])
    dx, dy, dz = dv

    X = (-Tz / dz) * dx + Tx
    Y = (-Tz / dz) * dy + Ty

    for num, camid in enumerate(cam_idxs):
        cam_tmp = cams[num]

        dmp = np.array(cam_tmp.get_img_corners()[0]) * 1000
        imgp, _ = cv.projectPoints(dmp, cam_tmp.rvecs, cam_tmp.tvecs, cam_tmp.mtx, None)

        for point in imgp.squeeze(1).astype(np.int32).tolist():
            cv.circle(frames[num], point, 15, (0, 0, 255), -1)

        point, _ = cv.projectPoints(
            np.array([[X, Y, 0]], dtype=np.float32),
            cam_tmp.rvecs,
            cam_tmp.tvecs,
            cam_tmp.mtx,
            None,
        )
        xr, yr = point[0][0].astype(np.int32)
        cv.circle(frames[num], (xr, yr), 24, (0, 255, 255), -1)

    frame1 = frames[camidxs[0]]
    frame1 = cv.rectangle(frame1, (0, 0), frame1.shape[:2][::-1], (0, 255, 0), 40)
    frame2 = frames[camidxs[1]]
    frame2 = cv.rectangle(frame2, (0, 0), frame2.shape[:2][::-1], (255, 0, 255), 40)
    screen = np.vstack([frame1, frame2])

    sidebar = []
    for idx, fr in enumerate(frames):
        if idx == camidxs[0] or idx == camidxs[1]:
            continue
        frame = fr.copy()
        frame = cv.rectangle(frame, (0, 0), frame.shape[:2][::-1], (255, 255, 255), 40)
        sidebar.append(cv.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4)))
    sidebar = sidebar[: len(frames) - 2]

    screen = np.hstack([screen, np.vstack(sidebar)])
    copy_frame = draw_field(screen.copy(), camid1, camid2)

    cv.imshow("frames", copy_frame)
    k = cv.waitKey(10)
    if k == ord("q"):
        break
    if k == ord("a"):
        camidxs[0] = (camidxs[0] + 1) % len(cam_idxs)
    if k == ord("d"):
        camidxs[1] = (camidxs[1] + 1) % len(cam_idxs)
