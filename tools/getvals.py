import numpy as np

lines = []

with open("detcs.txt", "r") as f:
    lines = f.readlines()

newlines = []
curridx = -1
for line in lines:

    el = list(map(int, line.strip().split(" ")))
    if el[0] != curridx:
        newlines.append([])
        curridx = el[0]

    newlines[curridx - 1].append(el[1:])

for i in range(len(newlines)):
    newlines[i] = np.array(newlines[i])


all_diff = []
all_areas = []
for cam in newlines:
    diffs = np.sqrt((cam[:-1, 0] - cam[1:, 0]) ** 2 + (cam[:-1, 1] - cam[1:, 1]) ** 2)
    diffs = diffs.astype(np.int32)
    all_diff += list(diffs)

    area = cam[:, 2] * cam[:, 3]
    all_areas += list(area)

all_diff = np.array(all_diff)
all_areas = np.array(all_areas)
all_diff.sort()
all_areas.sort()

print(f"Mean: {np.mean(all_diff)}, Std: {np.std(all_diff)}")
print(f"Mean: {np.mean(all_areas)}, Std: {np.std(all_areas)}")

detecs = {}


breakpoint()
