import sys

sys.path.append(".")

import os
import shutil
import random

from tqdm import tqdm

from setup import get_args_split

args = get_args_split()

DATASET_PATH = args["dataset"]
TARGET_PATH = args["target_dataset"]

if not os.path.exists(DATASET_PATH):
    print("Dataset not found")
    exit()
if os.path.exists(TARGET_PATH):
    print("Target dataset already exists")
    exit()
else:
    os.makedirs(TARGET_PATH)
    os.makedirs(TARGET_PATH + "/train")
    os.makedirs(TARGET_PATH + "/val")
    os.makedirs(TARGET_PATH + "/test")
    os.makedirs(TARGET_PATH + "/train/images")
    os.makedirs(TARGET_PATH + "/train/labels")
    os.makedirs(TARGET_PATH + "/val/images")
    os.makedirs(TARGET_PATH + "/val/labels")
    os.makedirs(TARGET_PATH + "/test/images")
    os.makedirs(TARGET_PATH + "/test/labels")

allfiles = os.listdir(DATASET_PATH + "/images")

trainlen = (len(allfiles) / 100) * 85
vallen = (len(allfiles) / 100) * 10
testlen = (len(allfiles) / 100) * 5

random.shuffle(allfiles)

train = allfiles[: int(trainlen)]
val = allfiles[int(trainlen) : int(trainlen + vallen)]
test = allfiles[int(trainlen + vallen) :]


for file in tqdm(train):
    shutil.copy(DATASET_PATH + "/images/" + file, TARGET_PATH + "/train/images")
    shutil.copy(
        DATASET_PATH + "/labels/" + file[:-4] + ".txt", TARGET_PATH + "/train/labels"
    )

for file in tqdm(test):
    shutil.copy(DATASET_PATH + "/images/" + file, TARGET_PATH + "/test/images")
    shutil.copy(
        DATASET_PATH + "/labels/" + file[:-4] + ".txt", TARGET_PATH + "/test/labels"
    )

for file in tqdm(val):
    shutil.copy(DATASET_PATH + "/images/" + file, TARGET_PATH + "/val/images")
    shutil.copy(
        DATASET_PATH + "/labels/" + file[:-4] + ".txt", TARGET_PATH + "/val/labels"
    )
