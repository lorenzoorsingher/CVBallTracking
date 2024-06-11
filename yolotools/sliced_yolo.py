import cv2 as cv
import torch

from random import randint
from ultralytics import YOLO


class SlicedYolo:
    def __init__(self, model_path="", wsize=(640, 640), overlap=(0, 0)):

        self.model_path = model_path
        self.wsize = wsize
        self.overlap = overlap
        self.model = YOLO(model_path)

    def print_windows(self, frame):
        imsize = frame.shape[:2][::-1]
        origins = self.get_windows(imsize)
        for ori in origins:
            x, y = ori
            cv.rectangle(
                frame,
                (x, y),
                (x + self.wsize[0], y + self.wsize[1]),
                (randint(100, 200), randint(100, 200), 0),
                15,
            )
        return frame

    def get_windows(self, imsize):
        window_size = self.wsize

        # print("window_size ", window_size)
        # print("imsize ", imsize)

        ox_size = int(window_size[0] * self.overlap[0])
        oy_size = int(window_size[1] * self.overlap[1])

        nx = (imsize[0] - window_size[0]) // (window_size[0] - ox_size) + 1
        ny = (imsize[1] - window_size[1]) // (window_size[1] - oy_size) + 1

        xpad = (imsize[0] - window_size[0]) % (window_size[0] - ox_size)
        ypad = (imsize[1] - window_size[1]) % (window_size[1] - oy_size)

        origins = []
        for i in range(nx):
            for j in range(ny):
                x = i * (window_size[0] - ox_size) + xpad // 2
                y = j * (window_size[1] - oy_size) + ypad // 2
                origins.append((x, y))
        return origins

    def predict(self, frame):

        cv.namedWindow("frame", cv.WINDOW_NORMAL)
        imsize = frame.shape[:2][::-1]
        origins = self.get_windows(imsize)

        # wframe = self.print_windows(frame.copy(), origins)

        windows = []
        for origin in origins:
            x, y = origin
            window = {
                "image": frame[
                    y : y + self.wsize[1],
                    x : x + self.wsize[0],
                ],
                "coo": (x, y, self.wsize[0], self.wsize[1]),
            }
            windows.append(window)

        detections = []
        for win in windows:

            img = win["image"]
            real_x, real_y, _, _ = win["coo"]
            result = self.model.predict(img, verbose=False)
            boxes = result[0].boxes.xywh.cpu().tolist()

            for idx, box in enumerate(boxes):
                conf = result[0].boxes.conf.tolist()[idx]
                x, y, w, h = map(int, box)
                detections.append((x + real_x, y + real_y, w, h, conf))

        if len(detections) == 0:
            return None, None
        else:
            out = detections[torch.argmax(torch.tensor(detections).T[-1]).item()]
            return out, detections
