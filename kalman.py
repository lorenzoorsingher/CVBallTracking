import cv2
import random
import numpy as np


class KalmanTracker:

    def __init__(self):

        self.started = False

        diff = 0.01

        ps = 0.5
        ms = 0.01

        print(f"ps: {ps} \t ms: {ms}")
        Hz = 25.0  # Frequency of Vision System
        dt = 1.0 / Hz
        v = dt
        a = 0.5 * (dt**2)
        self.kalman = cv2.KalmanFilter(9, 3, 0)

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, v, 0, 0, a, 0, 0],
                [0, 1, 0, 0, v, 0, 0, a, 0],
                [0, 0, 1, 0, 0, v, 0, 0, a],
            ],
            np.float32,
        )

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, v, 0, 0, a, 0, 0],
                [0, 1, 0, 0, v, 0, 0, a, 0],
                [0, 0, 1, 0, 0, v, 0, 0, a],
                [0, 0, 0, 1, 0, 0, v, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, v, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, v],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        self.kalman.processNoiseCov = (
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                np.float32,
            )
            * ps
        )

        self.kalman.measurementNoiseCov = (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * ms
        )

    def update(self, point):

        red_point = point / 1000

        if not self.started:
            self.kalman.statePre = np.array(
                [
                    [np.float32(red_point[0] + random.randint(-7, 7) / 10)],
                    [np.float32(red_point[1] + random.randint(-7, 7) / 10)],
                    [np.float32(red_point[2] + random.randint(-7, 7) / 10)],
                    [np.float32(0.0)],
                    [np.float32(0.0)],
                    [np.float32(0.0)],
                    [np.float32(0.0)],
                    [np.float32(0.0)],
                    [np.float32(0.0)],
                ]
            )
        mp = np.array(
            [
                [np.float32(red_point[0])],
                [np.float32(red_point[1])],
                [np.float32(red_point[2])],
            ]
        )
        self.kalman.correct(mp)
        tp = self.kalman.predict()

        tp *= 1000
        return tp
