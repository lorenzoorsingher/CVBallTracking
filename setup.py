import argparse


def get_args_corners():
    parser = argparse.ArgumentParser(
        prog="corners_detection.py",
        description="""Runs corner detection""",
    )

    parser.add_argument(
        "-cs",
        "--cameras",
        type=str,
        help="Set the indexes of the cameras you want to calibrate separated by comma",
        default="-1",
        metavar="",
    )

    parser.add_argument(
        "-dn",
        "--detect-num",
        type=int,
        help="Minimum number of detections to stop the process [20]",
        default=20,
        metavar="",
    )

    parser.add_argument(
        "-dt",
        "--distance-threshold",
        type=int,
        help="Distance threshold between detections [100]",
        default=100,
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_calib():
    parser = argparse.ArgumentParser(
        prog="calibrate_cameras.py",
        description="""Run camera calibration""",
    )

    parser.add_argument(
        "-cs",
        "--cameras",
        type=str,
        help="Set the indexes of the cameras you want to calibrate separated by comma",
        default="-1",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_agument():
    parser = argparse.ArgumentParser(
        prog="agument.py",
        description="""Agument dataset""",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset path",
        default="yolotools/datasets/overfitter_1",
        metavar="",
    )

    parser.add_argument(
        "-t",
        "--target_dataset",
        type=str,
        help="Target dataset path",
        default="yolotools/datasets/overfitter_mega",
        metavar="",
    )

    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        help="Set the amount of samples to generate",
        default=4000,
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_split():
    parser = argparse.ArgumentParser(
        prog="split.py",
        description="""Splits dataset in train test val""",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset path",
        default="yolotools/datasets/overfitter_mega",
        metavar="",
    )

    parser.add_argument(
        "-t",
        "--target_dataset",
        type=str,
        help="Target dataset path",
        default="yolotools/datasets/overfitter_mega_split",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_checklabels():
    parser = argparse.ArgumentParser(
        prog="checklabels.py",
        description="""Check YOLO labels""",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset path",
        default="yolotools/datasets/overfitter_mega",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def get_args_demo():
    parser = argparse.ArgumentParser(
        prog="multi_track_demo.py",
        description="""Multi camera tracking demo""",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        help="Performance mode [higher -> more precise & slower]",
        default=2,
        metavar="",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Starting frame",
        default=750,
        metavar="",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="End frame",
        default=1000,
        metavar="",
    )

    parser.add_argument(
        "-F",
        "--from_file",
        action="store_true",
        default=False,
        help="Use detections from file",
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
        default=-1,
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
