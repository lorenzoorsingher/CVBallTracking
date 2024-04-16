import os
import json
import numpy as np
import cv2 as cv

from time import strftime, localtime, time

from common import NumpyEncoder


class CameraController:
    def __init__(self, index, path="data/camera_data/"):

        self.mtx = None
        self.dist = None
        self.rvecs = np.array([[0], [0], [0]], dtype=np.float32)
        self.tvecs = np.array([[0], [0], [0]], dtype=np.float32)
        self.index = index

        self.main_path = f"{path}cam_{index}/"
        self.meta_file = f"{path}cam_{index}/metadata.json"
        self.dump_path = f"{self.main_path}dump/"
        self.calib_path = f"{self.main_path}calib/"

        self.imsize = None
        self.chessboard_size = None
        self.cell_size = 28

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

    def load_params(self, dump_file="latest"):

        calib_file = f"{self.calib_path}camera_calib.json"
        if not os.path.exists(calib_file):
            print(f"[camera] No calib files found for camera {self.index}, skipping...")
        else:

            with open(calib_file, "r") as f:
                cal = json.loads(f.read())
                self.mtx = np.array(cal["mtx"], dtype=np.float32)
                self.dist = np.array(cal["dist"], dtype=np.float32)
                self.rvecs = np.array(cal["rvecs"], dtype=np.float32)
                self.tvecs = np.array(cal["tvecs"], dtype=np.float32)

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

    def save_calib(self, mtx=None, dist=None, rvecs=None, tvecs=None):

        print("[Calibration] Saving calibration for camera ", self.index)

        if mtx is None or dist is None:
            mtx = self.mtx
            dist = self.dist
        else:
            self.mtx = mtx
            self.dist = dist
        # TODO check if loading is correct when tvecs/rvecs are None
        if rvecs is None or tvecs is None:
            rvecs = self.rvecs
            tvecs = self.tvecs
        else:
            self.rvecs = rvecs
            self.tvecs = tvecs

        with open(f"{self.calib_path}camera_calib.json", "w") as f:
            json.dump(
                {
                    "mtx": mtx.tolist(),
                    "dist": dist.tolist(),
                    "tvecs": tvecs.tolist(),
                    "rvecs": rvecs.tolist(),
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

        objp = objp * self.cell_size

        return objp

    def save_img_corners(self, real_corners, img_corners):

        with open(f"{self.calib_path}img_points.json", "w") as f:
            json.dump(
                {
                    "real_corners": real_corners,
                    "img_corners": img_corners,
                },
                f,
            )

    def get_img_corners(self):

        if not os.path.exists(f"{self.calib_path}img_points.json"):
            print("[camera] No dump img corners file found, exiting...")
            exit()

        with open(f"{self.calib_path}img_points.json", "r") as f:
            dmp = json.loads(f.read())
        return dmp["real_corners"], dmp["img_corners"]

    ###########################################################################

    def get_camera_position(self):
        rotation_matrix, _ = cv.Rodrigues(self.rvecs)
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        inv_tvecs = -np.dot(inverse_rotation_matrix, self.tvecs)
        return inverse_rotation_matrix, inv_tvecs

    def estimate_camera_position(self, real_corners, img_corners):
        real_corners = np.array(real_corners, dtype=np.float32)
        img_corners = np.array(img_corners, dtype=np.float32)

        ret, rvecs, tvecs = cv.solvePnP(
            np.array(real_corners, dtype=np.float32),
            np.array(img_corners, dtype=np.float32),
            self.mtx,
            self.dist,
        )

        self.rvecs = rvecs
        self.tvecs = tvecs

        self.save_calib(rvecs=rvecs, tvecs=tvecs)

        return self.get_camera_position()

    def repoject_points(self, objp):
        imgp, _ = cv.projectPoints(objp, self.rvecs, self.tvecs, self.mtx, self.dist)
        return imgp
