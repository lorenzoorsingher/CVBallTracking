from ultralytics import YOLO
import cv2 as cv
import numpy as np


class SlicedYolo:
    def __init__(self, model_path="", grid=(-1, -1), wsize=(640, 640)):

        self.model_path = model_path
        self.grid = grid
        self.wsize = wsize
        self.model = YOLO(model_path)

    def predict(self, frame):
        cv.namedWindow("frame", cv.WINDOW_NORMAL)
        imsize = frame.shape[:2][::-1]
        if self.grid[0] == -1:
            window_size = self.wsize
        else:
            window_size = (
                imsize[0] // self.grid[0],
                imsize[1] // self.grid[1],
            )

        print("window_size ", window_size)
        print("imsize ", imsize)
        nx = imsize[0] // window_size[0]
        ny = imsize[1] // window_size[1]
        xpad = imsize[0] - nx * window_size[0]
        ypad = imsize[1] - ny * window_size[1]

        windows = []
        for i in range(nx):
            for j in range(ny):
                x = i * window_size[0] + xpad
                y = j * window_size[1] + ypad
                cv.rectangle(
                    frame,
                    (x, y),
                    (x + window_size[0], y + window_size[1]),
                    (0, 255, 0),
                    2,
                )
                window = {
                    "image": frame[
                        y : y + window_size[1],
                        x : x + window_size[0],
                    ],
                    "coo": (x, y, window_size[0], window_size[1]),
                }
                windows.append(window)

        new_frame = np.zeros_like(frame)

        for win in windows:
            x, y, xsize, ysize = win["coo"]
            new_frame[
                y : y + ysize,
                x : x + xsize,
            ] = win["image"]

        cv.imshow("frame", new_frame)
        cv.waitKey(0)
        # result = self.model.predict(frame)
        # return result
