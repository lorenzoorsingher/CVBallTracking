import os
import json
import numpy as np
import cv2 as cv

from time import strftime, localtime, time

from common import NumpyEncoder


class CameraController:
    def __init__(self, index, path="data/camera_data/", imsize=(3840, 2160)):
        """
        This class is used to control camera data, such as calibration parameters,
        it manages the files and directories for each camera and saves and loads
        calibration parameters as well as some logic for camera position estimation

        Parameters:
        - index (int): Camera index
        - path (str): Path to camera data directory
        - imsize (tuple): Image size of camera
        """
        self.mtx = None
        self.dist = None
        self.rvecs = np.array([[0], [0], [0]], dtype=np.float32)
        self.tvecs = np.array([[0], [0], [0]], dtype=np.float32)
        self.index = index

        self.main_path = f"{path}cam_{index}/"
        self.meta_file = f"{path}cam_{index}/metadata.json"
        self.dump_path = f"{self.main_path}dump/"
        self.calib_path = f"{self.main_path}calib/"

        self.imsize = imsize
        self.chessboard_size = None
        self.cell_size = 28

        self.build_tree()
        self.load_params()

    def build_tree(self):
        """
        Create directories for camera data if they do not exist
        """
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        if not os.path.exists(self.calib_path):
            os.makedirs(self.calib_path)

    def load_params(self):
        """
        Load calibration parameters from file and
        load metadata from file as well as chessboard size
        """
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

    def save_dump(self, all_corners):
        """
        Save dump of all detected corners to file

        Parameters:
        - all_corners (np.array): Array of all detected corners
        """
        file_name = strftime("dump_%Y%m%d_%H%M%S.json", localtime(time()))

        with open(f"{self.dump_path}{file_name}", "w") as f:
            json.dump(all_corners, f, cls=NumpyEncoder)

    def get_dump(self, dump_file="latest"):
        """
        Returns the dump of all detected corners, if no dump file is specified
        the latest dump file will be loaded

        Parameters:
        - dump_file (str): Name of the dump file to load

        Returns:
        - np.array: Array of all detected corners
        """

        if len(os.listdir(self.dump_path)) == 0:
            print("[camera] No dump files found...")
            return None
        else:
            if dump_file == "latest":
                dump_file = self.dump_path + sorted(os.listdir(self.dump_path))[-1]

        with open(dump_file, "r") as f:
            dmp = np.array(json.loads(f.read()), dtype=np.float32)
        return dmp

    def save_calib(self, mtx=None, dist=None, rvecs=None, tvecs=None):
        """
        Save calibration parameters to file, if no parameters are specified
        the current parameters will be saved

        Parameters:
        - mtx (np.array): Camera matrix
        - dist (np.array): Distortion coefficients
        - rvecs (np.array): Rotation vectors
        - tvecs (np.array): Translation vectors
        """
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
        """
        Generate chessboard points based on the chessboard size and cell size

        Returns:
        - np.array: Array of chessboard points
        """
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)

        objp = objp * self.cell_size

        return objp

    def save_img_corners(self, real_corners, img_corners):
        """
        Save the real and image corners to file

        Parameters:
        - real_corners (np.array): Real corners
        - img_corners (np.array): Image corners
        """
        with open(f"{self.calib_path}img_points.json", "w") as f:
            json.dump(
                {
                    "real_corners": real_corners,
                    "img_corners": img_corners,
                },
                f,
            )

    def get_img_corners(self):
        """
        Returns the real and image corners from file

        Returns:
        - np.array: Real corners
        - np.array: Image corners
        """
        if not os.path.exists(f"{self.calib_path}img_points.json"):
            print("[camera] No dump img corners file found, exiting...")
            exit()

        with open(f"{self.calib_path}img_points.json", "r") as f:
            dmp = json.loads(f.read())
        return dmp["real_corners"], dmp["img_corners"]

    def get_camera_position(self):
        """
        Inverts tvecs and rvecs to get the camera position in world coordinates

        Returns:
        - np.array: Rotation matrix
        - np.array: Translation vectors
        """

        rot_mtx, _ = cv.Rodrigues(self.rvecs)
        world_rot_mtx = np.linalg.inv(rot_mtx)
        world_tvecs = -np.dot(world_rot_mtx, self.tvecs)
        return world_rot_mtx, world_tvecs

    def estimate_camera_position(self, real_corners, img_corners):
        """
        Estimate the camera position based on real and image corners using
        solvePnP and save the calibration results to file. Returns the camera
        position in world coordinates

        Parameters:
        - real_corners (np.array): Real corners
        - img_corners (np.array): Image corners

        Returns:
        - np.array: Rotation matrix
        - np.array: Translation vectors
        """
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

    def undistort_img(self, img):
        """
        Undistort image using camera matrix and distortion coefficients

        Parameters:
        - img (np.array): Image to undistort

        Returns:
        - np.array: Undistorted image
        """
        return cv.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_projection_matrix(self):
        """
        Get the projection matrix of the camera

        Returns:
        - np.array: Projection matrix
        """

        rot_mtx, _ = cv.Rodrigues(self.rvecs)
        # T = np.eye(4, dtype=np.float64)
        # T[:3, :3] = rot_mtx
        # T[:3, 3] = self.tvecs.T
        # breakpoint()
        # # np.dot(self.mtx, np.hstack((rot_mtx, self.tvecs)))
        return np.dot(self.mtx, np.hstack((rot_mtx, self.tvecs)))

    def triangulate(self, cam2, point2d1, point2d2):
        proj1 = self.get_projection_matrix()
        proj2 = cam2.get_projection_matrix()

        point2d1 = np.array([point2d1], dtype=np.float32)
        point2d2 = np.array([point2d2], dtype=np.float32)

        point4d = cv.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)
        point3d = cv.convertPointsFromHomogeneous(point4d.T)[0][0]
        return point3d
