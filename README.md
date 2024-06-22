
# FAH (Funny Achronym Here)

Project for Computer Vision, UniTN

<div style="text-align:center;">
<img src="imgs/3dplot.png" width=60%>

</div>

## About
The goal of the project is to build a system that can track a ball in a 3D space using multiple cameras. Our objective is to develop a tool able to make the setup of the cameras easy, fast and precise, and to provide a simple and intuitive interface to the user. 

<br>
<div style="text-align:center;">
<img src="imgs/calibration.png" width=60%>
</div>
<br>

## General Information



### Project Structure

```

CVBallTracking
├── data
│   ├── camera_data
│   │   ├── cam_1
│   │   │   ├── calib
│   │   │   │   ├── camera_calib.json
│   │   │   │   └── img_points.json
│   │   │   ├── dump
│   │   │   │   ├── dump_20240408_125237.json
│   │   │   │   └── ...
│   │   │   └── metadata.json
│   │   ├── cam_2
│   │   │   └── ...
│   │   ├── camera_positions.json
│   │   └── chess_sizes.json
│   └── video
│       ├── out1F.mp4
│       └── ...
├── weights
│   └── best.pt
├── yolotools
│   ├── datasets
│   │   └── ...
│   ├── agument_diy.py
│   ├── annotator.py
│   ├── checklabels.py
│   ├── data.yaml
│   ├── sliced_yolo.py
│   ├── split.py
│   ├── test_multicam.py
│   ├── train.ipynb
│   └── train.py
├── 3dplot.py
├── build_map.py
├── calibrate_cameras.py
├── calib_test.py
├── camera_controller.py
├── common.py
├── corners_detection.py
├── kalman.py
├── multiple_views.py
├── multi_track_demo.py
├── pose_estimation.py
├── README.md
├── setup.py
└── sort.py

```


## Usage



## Calibration



## Testing



## TODO 

- comparative testing sys
- class-ify calibration scripts
- clean up code
- report



## Contacts 

- [@lorenzoorsingher](https://github.com/lorenzoorsingher)
- [@AlessiaPivotto](https://github.com/AlessiaPivotto)
- [GitHub repo](https://github.com/lorenzoorsingher/CVBallTracking)
