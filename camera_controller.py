import os
import json
import numpy as np

from time import strftime, localtime, time

from common import NumpyEncoder


class CameraController:
    def __init__(self, index, path="data/camera_data/"):

        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.index = index
        self.main_path = f"{path}cam_{index}/"
        self.dump_path = f"{self.main_path}dump/"
        self.calib_path = f"{self.main_path}calib/"

    def build_tree(self):
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        if not os.path.exists(self.calib_path):
            os.makedirs(self.calib_path)

    def load_params(self, calib_file="latest", dump_file="latest"):

        if calib_file == "latest":
            calib_file = self.calib_path + sorted(os.listdir(self.calib_path))[-1]
        if dump_file == "latest":
            dump_file = self.dump_path + sorted(os.listdir(self.dump_path))[-1]

        dmp = np.array(json.loads(open(dump_file).read()), dtype=np.float32)
        breakpoint()

    def save_dump(self, all_corners):

        file_name = strftime("dump_%Y%m%d_%H%M%S.json", localtime(time()))

        with open(f"{self.dump_path}{file_name}", "w") as f:
            json.dump(all_corners, f, cls=NumpyEncoder)
