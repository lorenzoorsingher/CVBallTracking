
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
│   ├── camera_data
│   │   ├── cam_1
│   │   │   ├── calib
│   │   │   │   ├── camera_calib.json
│   │   │   │   └── img_points.json
│   │   │   ├── dump
│   │   │   │   ├── dump_20240408_125237.json
│   │   │   │   └── ...
│   │   │   └── metadata.json
│   │   ├── ...
│   │   ├── camera_positions.json
│   │   └── chess_sizes.json
│   ├── video
│   │   ├── out1F.mp4
│   │   └── ...
│   └── data.placeholder
├── imgs
│   └── ...
├── 3dplot.py
├── calibrate_cameras.py
├── camera_controller.py
├── common.py
├── corners_detection.py
├── multiple_views.py
├── pose_estimation.py
├── README.md
├── requirements.txt
├── setup.py
└── utils.py

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
