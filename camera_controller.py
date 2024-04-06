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
        self.meta_file = f"{path}cam_{index}/metadata.json"
        self.dump_path = f"{self.main_path}dump/"
        self.calib_path = f"{self.main_path}calib/"

        self.imsize = None
        self.chessboard_size = None

        self.build_tree()
        self.load_params()

    def build_tree(self):
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        if not os.path.exists(self.calib_path):
            os.makedirs(self.calib_path)
        if not os.path.exists(self.meta_file):
            with open(self.meta_file, "w") as f:
                json.dump({"imsize": (0, 0)}, f)

    def load_params(self, calib_file="latest", dump_file="latest"):

        if len(os.listdir(self.calib_path)) == 0:
            print("[camera] No calibration files found, skipping...")
        else:
            if calib_file == "latest":
                calib_file = self.calib_path + sorted(os.listdir(self.calib_path))[-1]

        if len(os.listdir(self.dump_path)) == 0:
            print("[camera] No dump files found, skipping...")
        else:
            if dump_file == "latest":
                dump_file = self.dump_path + sorted(os.listdir(self.dump_path))[-1]

        metadata = json.loads(open(self.meta_file).read())
        self.imsize = metadata["imsize"]

        sizes_path = "data/camera_data/chess_sizes.json"
        with open(sizes_path, "r") as file:
            sizes = json.load(file)
        self.chessboard_size = sizes[str(self.index)]

    def save_dump(self, all_corners, imsize):

        file_name = strftime("dump_%Y%m%d_%H%M%S.json", localtime(time()))

        with open(f"{self.dump_path}{file_name}", "w") as f:
            json.dump(all_corners, f, cls=NumpyEncoder)

        with open(self.meta_file, "w") as f:
            json.dump({"imsize": imsize}, f)

    def get_dump(self, dump_file="latest"):

        if len(os.listdir(self.dump_path)) == 0:
            print("[camera] No dump files found, exiting...")
            exit()
        else:
            if dump_file == "latest":
                dump_file = self.dump_path + sorted(os.listdir(self.dump_path))[-1]

        with open(dump_file, "r") as f:
            dmp = np.array(json.loads(f.read()), dtype=np.float32)
        return dmp

    def save_calib(self, mtx, dist):

        file_name = strftime("calib_%Y%m%d_%H%M%S.json", localtime(time()))

        self.mtx = mtx
        self.dist = dist

        with open(f"{self.calib_path}{file_name}", "w") as f:
            json.dump(
                {
                    "mtx": mtx.tolist(),
                    "dist": dist.tolist(),
                },
                f,
            )

    def get_chessboard(self):
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)

        return objp
