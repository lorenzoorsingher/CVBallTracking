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

    # parser.add_argument(
    #     "-s",
    #     "--sequence",
    #     type=int,
    #     help="Index of KITTI sequence [kitti mode]",
    #     default=0,
    #     metavar="",
    # )

    args = vars(parser.parse_args())
    return args


ù


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

    # parser.add_argument(
    #     "-s",
    #     "--sequence",
    #     type=int,
    #     help="Index of KITTI sequence [kitti mode]",
    #     default=0,
    #     metavar="",
    # )

    args = vars(parser.parse_args())
    return args
