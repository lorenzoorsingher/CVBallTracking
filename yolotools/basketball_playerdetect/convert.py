import os
import json
import numpy as np
import cv2 as cv


def visibile(point, frame):
    x, y = point
    if x < 0 or y < 0 or x > frame.shape[1] or y > frame.shape[0]:
        return False
    return True


dataset_path = "datasets/baskingball"

json_path = os.path.join(dataset_path, "deepsport_dataset_dataset.json")

resolutions = {}

# load json
with open(json_path, "r") as f:
    data = json.load(f)

cv.namedWindow("track", cv.WINDOW_NORMAL)

for lab in data:
    arena_label = lab["arena_label"]
    game_id = lab["game_id"]
    ts = lab["timestamp"]

    game_path = os.path.join(dataset_path, arena_label, str(game_id))

    # cam2_path = f"{game_path}/camcourt2_{ts}.json"

    # with open(cam2_path, "r") as f:
    #     cam2_data = json.load(f)

    for i in [1, 2]:

        labels = []
        cam_path = f"{game_path}/camcourt{i}_{ts}.json"
        with open(cam_path, "r") as f:
            cam_data = json.load(f)

        mtx = np.array(cam_data["calibration"]["KK"]).reshape(3, 3)
        R = np.array(cam_data["calibration"]["R"]).reshape(3, 3)
        t = np.array(cam_data["calibration"]["T"])
        kc = np.array(cam_data["calibration"]["kc"])
        impath = f"{game_path}/camcourt{i}_{ts}_0.png"
        frame = cv.imread(impath)
        og = frame.copy()
        annot = lab["annotations"]

        for ann in annot:

            if ann["type"] == "ball" and ann["visible"] == True:
                x, y, z = ann["center"]

                # project 3d point to 2d
                points_3d = np.array([x, y, z])
                points_2d, _ = cv.projectPoints(points_3d, R, t, mtx, kc)
                points_2d = points_2d[0][0]
                if visibile(points_2d, frame):

                    # ballsize = 30
                    balls_x = 0.019  # ballsize / frame.shape[1]
                    balls_y = 0.025  # ballsize / frame.shape[0]
                    x, y = points_2d
                    x_norm = x / frame.shape[1]
                    y_norm = y / frame.shape[0]

                    w = balls_x
                    h = balls_y
                    labels.append([0, x_norm, y_norm, w, h])
                    # w =
                    # print(balls_x, balls_y)
            elif ann["type"] == "player":
                all_points = []
                for key, val in ann.items():

                    if type(val) == list:

                        points_3d = np.array(val)
                        points_2d, _ = cv.projectPoints(points_3d, R, t, mtx, kc)
                        points_2d = points_2d[0][0]
                        all_points.append(points_2d)
                all_points = np.array(all_points)

                xmax = np.max(all_points[:, 0])
                xmin = np.min(all_points[:, 0])
                ymax = np.max(all_points[:, 1])
                ymin = np.min(all_points[:, 1])

                if visibile((xmin, ymin), frame) and visibile((xmax, ymax), frame):

                    x_norm = ((xmin + xmax) / 2) / frame.shape[1]
                    y_norm = ((ymin + ymax) / 2) / frame.shape[0]
                    w = ((xmax - xmin) / frame.shape[1]) + 30 / frame.shape[1]
                    h = ((ymax - ymin) / frame.shape[0]) + 30 / frame.shape[0]

                    if x_norm < 0 or y_norm < 0 or w < 0 or h < 0:
                        continue
                    elif x_norm > 1 or y_norm > 1 or x_norm + w > 1 or y_norm + h > 1:
                        continue
                    else:

                        # print(x_norm, y_norm, w, h)
                        labels.append([1, x_norm, y_norm, w, h])

        # print(labels)

        for label in labels:
            x, y, w, h = label[1:]
            x = int(x * frame.shape[1])
            y = int(y * frame.shape[0])
            w = int(w * frame.shape[1])
            h = int(h * frame.shape[0])

            if label[0] == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv.rectangle(
                frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                color,
                2,
            )

        dest_path = "datasets/ayy"

        imgs_path = os.path.join(dest_path, "images")
        labels_path = os.path.join(dest_path, "labels")

        newname = f"{arena_label}_{game_id}_{ts}_cam{i}"

        resized = cv.resize(og, (int(frame.shape[1] * (1752 / frame.shape[0])), 1752))
        resolutions[resized.shape] = 1
        cv.imwrite(f"{imgs_path}/{newname}.png", resized)
        with open(f"{labels_path}/{newname}.txt", "w") as f:
            for label in labels:
                f.write(f"{' '.join([str(x) for x in label])}\n")
        print(resolutions)
        cv.imshow("track", frame)
        cv.waitKey(1)
        # breakpoint()
    # breakpoint()
