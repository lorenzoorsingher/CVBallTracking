import albumentations as A
import os
import cv2 as cv
import numpy as np
import random
from tqdm import tqdm


def yolo_to_alb(label):
    if len(label) == 0:
        return []
    new_label = label[1:] + [label[0]]
    return new_label


def alb_to_yolo(label):

    if len(label) == 0:
        return []
    label = list(label)
    new_label = [label[-1]] + label[:-1]
    return new_label


transform = A.Compose(
    [
        A.RandomCrop(width=640, height=640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo"),
)


dataset_path = (
    "/home/lollo/Documents/python/CV/CVBallTracking/yolotools/datasets/overfitter_1"
)

target_ds = (
    "/home/lollo/Documents/python/CV/CVBallTracking/yolotools/datasets/overfitter_mega"
)
target_imgs = os.path.join(target_ds, "images")
target_labels = os.path.join(target_ds, "labels")

# folder = os.path.join(dataset_path, "test")
folder = dataset_path
imgs_path = os.path.join(folder, "images")
labels_path = os.path.join(folder, "labels")

cv.namedWindow("img", cv.WINDOW_NORMAL)

ball = 0
empty = 0


all_boxes = []
all_names = []
img_paths = []
breaker = 0
for file in tqdm(os.listdir(imgs_path)):
    img_path = os.path.join(imgs_path, file)
    label_path = os.path.join(labels_path, file[:-4] + ".txt")
    all_names.append(file[:-4])

    with open(label_path, "r") as f:
        line = f.readlines()[0]
        c, x, y, w, h = map(float, line.split(" "))
        box = yolo_to_alb([int(c), x, y, w, h])
    img_paths.append(img_path)
    all_boxes.append(box)

    breaker += 1
    # if breaker == 30:
    #     break

images_fin = []
labels_fin = []
names_fin = []
idxs = list(range(len(img_paths)))
# idxs = np.random.permutation(idxs)

TARGET = 4000
while True:

    idx = idxs[random.randint(0, len(idxs) - 1)]
    print(idx)
    # img = all_imgs[idx]
    img = cv.imread(img_paths[idx])

    box = all_boxes[idx]
    name = all_names[idx]

    #######################################################
    if ball < TARGET:
        fball = False
        while not fball:
            try:
                transformed = transform(image=img, bboxes=[box])
            except:
                print("error")
                continue
            transformed_image = transformed["image"]
            show_image = transformed_image.copy()
            transformed_bboxes = transformed["bboxes"]
            # print(transformed_bboxes)

            yolo_bboxes = []
            for bbox in transformed_bboxes:
                yolo_bboxes.append(alb_to_yolo(bbox))

            for bbox in yolo_bboxes:
                # print(bbox)
                c, x, y, w, h = bbox
                if c == 0:
                    x = int(x * show_image.shape[1])
                    y = int(y * show_image.shape[0])
                    w = int(w * show_image.shape[1])
                    h = int(h * show_image.shape[0])
                    cv.rectangle(
                        show_image,
                        (x - w // 2, y - h // 2),
                        (x + w // 2, y + h // 2),
                        (255, 255, 0),
                        2,
                    )
                    fball = True
                    ball += 1
            ######################################################
    else:
        try:
            transformed = transform(image=img, bboxes=[box])
        except:
            print("error")
            continue
        transformed_image = transformed["image"]
        show_image = transformed_image.copy()
        transformed_bboxes = transformed["bboxes"]
        # print(transformed_bboxes)

        yolo_bboxes = []
        for bbox in transformed_bboxes:
            yolo_bboxes.append(alb_to_yolo(bbox))
        empty += 1

    images_fin.append(transformed_image)
    labels_fin.append(yolo_bboxes)
    names_fin.append(name)

    if ball >= TARGET and empty >= TARGET:
        break
    print("-------------------------")
    print(f"ball: {ball}\nempty: {empty}")
    print(f"found ball: {fball}\nfound empty: {yolo_bboxes == []}")
    cv.imshow("img", show_image)
    cv.waitKey(1)


print(f"Final number of images: {len(images_fin)}")


for idx in tqdm(range(len(images_fin))):
    img = images_fin[idx]
    lines = labels_fin[idx]
    name = names_fin[idx]

    cv.imwrite(f"{target_imgs}/{idx}_{name}.png", img)
    with open(f"{target_labels}/{idx}_{name}.txt", "w") as f:
        for label in lines:
            f.write(f"{' '.join([str(x) for x in label])}\n")
