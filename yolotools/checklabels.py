import os
import cv2 as cv
import numpy as np


dataset_path = "/home/lollo/Documents/python/CV/CVBallTracking/yolotools/datasets/fakebasket_20240610181236"

folder = os.path.join(dataset_path, "test")
folder = dataset_path
imgs_path = os.path.join(folder, "images")
labels_path = os.path.join(folder, "labels")

cv.namedWindow("img", cv.WINDOW_NORMAL)

for file in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, file)
    label_path = os.path.join(labels_path, file[:-4] + ".txt")

    img = cv.imread(img_path)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip().split(" ")
        print(line)
        x, y, w, h = map(float, line[1:5])

        x = int(x * img.shape[1])
        y = int(y * img.shape[0])
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])
        cv.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 255, 0), 2
        )

    cv.imshow("img", img)
    cv.waitKey(0)
