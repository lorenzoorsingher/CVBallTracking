import os
import sys
import shutil
import random

ogpath = "/home/lollo/Documents/python/yolo/datasets/amogus"
dest_path = "/home/lollo/Documents/python/yolo/datasets/mogusBig"


allfiles = os.listdir(ogpath + "/images")

trainlen = (len(allfiles) / 100) * 85
vallen = (len(allfiles) / 100) * 10
testlen = (len(allfiles) / 100) * 5

random.shuffle(allfiles)

train = allfiles[: int(trainlen)]
val = allfiles[int(trainlen) : int(trainlen + vallen)]
test = allfiles[int(trainlen + vallen) :]


for file in train:
    shutil.copy(ogpath + "/images/" + file, dest_path + "/train/images")
    shutil.copy(ogpath + "/labels/" + file[:-4] + ".txt", dest_path + "/train/labels")

for file in test:
    shutil.copy(ogpath + "/images/" + file, dest_path + "/test/images")
    shutil.copy(ogpath + "/labels/" + file[:-4] + ".txt", dest_path + "/test/labels")

for file in val:
    shutil.copy(ogpath + "/images/" + file, dest_path + "/val/images")
    shutil.copy(ogpath + "/labels/" + file[:-4] + ".txt", dest_path + "/val/labels")

breakpoint()
