import argparse


def get_args_corners():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Run camera calibration""",
    )

    parser.add_argument(
        "-cs",
        "--cameras",
        type=str,
        help="Set the indexes of the cameras you want to calibrate separated by comma",
        default="1",
        metavar="",
    )

    parser.add_argument(
        "-dn",
        "--detect-num",
        type=int,
        help="Minimum number of detections to stop the process",
        default=20,
        metavar="",
    )

    parser.add_argument(
        "-dt",
        "--distance-threshold",
        type=int,
        help="Minimum number of detections to stop the process",
        default=100,
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_calib():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Run camera calibration""",
    )

    parser.add_argument(
        "-cs",
        "--cameras",
        type=str,
        help="Set the indexes of the cameras you want to calibrate separated by comma",
        default="1",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_pose():
    parser = argparse.ArgumentParser(
        prog="pose_estimation.py",
        description="""Run pose estimation""",
    )

    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        help="Set the index of the camera you want to calibrate",
        default=1,
        metavar="",
    )

    parser.add_argument(
        "-R",
        "--reuse",
        action="store_true",
        help="Reuse old screen point data",
    )

    args = vars(parser.parse_args())
    return args