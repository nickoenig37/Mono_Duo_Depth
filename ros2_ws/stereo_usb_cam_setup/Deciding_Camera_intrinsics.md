# This doc is just keeping track of the camera instrinsics I decided on after testing.

## Calibrating the cameras:
Needed to install:
```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

I am running this script:

```bash
ros2 run camera_calibration cameracalibrator --size 7x7 --square 0.025 --approximate 0.1 \
--ros-args \
-r right:=/camera/right/image_raw \
-r left:=/camera/left/image_raw \
-r right_camera:=/camera/right \
-r left_camera:=/camera/left
```

- Then the window will open, move your chessboard in front of both cameras until you get enough samples (green).
- Then click "CALIBRATE" and after it finishes click "SAVE" to save the calibration files to the appropriate folders.



## Notes on the calibration and how this is going to work:
To make your "Sensor Swap" work, you must force both the Training Set and the Robot Set to be rectified into the same virtual optical model

If you use standard cv2.stereoRectify separately for both, OpenCV will auto-crop and zoom differently for each pair, resulting in different focal lengths. The network will learn scale "A" during training and see scale "B" during testing, leading to bad depth estimation.
The Fix: Manually define a Target Camera Matrix ($P_{target}$) and force both rectifications to use it.

## Doing calibration between the realsense and the right camera:
We need to do this ourselves since we're calculating between the realsense and the right usb so when we train we know the exact relationship between the two cameras.
I used a 8x8 25mm checkerboard pattern printed on paper.
* I then ran the capture_calibration.py script to capture images from both cameras simultaneously.
- When doing this you need to click on the image window that pops up and you click s to save an image pair-- your images need to do the following:
Required Poses (aim for 30–40 pairs total):
```bash
The "XY" Fill (Coverage):

Center: Hold the board in the dead center.

Corners: Move the board to the extreme Top-Left, Top-Right, Bottom-Left, and Bottom-Right.

Why? This maps the lens distortion. Lenses are most distorted at the edges. If you only capture the center, the network will fail when an object moves to the side of the screen.

The "Skew" (Depth Perception):

Tilt Back: Angle the top of the board away from the camera (approx 30-45 degrees).

Tilt Forward: Angle the bottom of the board away.

Turn Left: Angle the left side away.

Turn Right: Angle the right side away.

Why? This is the most critical step for calculating Focal Length. If the board is always flat facing the camera, the math cannot distinguish between "a small board close up" and "a big board far away." Tilting it forces the perspective change that solves this ambiguity.
```

#### After capturing enough image pairs, run the calibrate_realsense_to_right.py script to compute the calibration:
```bash
python3 calibrate_realsense_to_right.py
```
- Then evaluate the results printed out 

RESULTS OF THE CALIBRATION FOR THE REALSENSE TO THE USBCAM RIGHT CAMERA:
```
--- RESULTS ---
RMS Error: 16.251240759472633

Left Camera Matrix (K1):
 [[342.38669154   0.         315.93207959]
 [  0.         356.2293514  210.71972505]
 [  0.           0.           1.        ]]

Right Camera Matrix (K2):
 [[269.17348427   0.         338.58500081]
 [  0.         294.73748816 231.62601469]
 [  0.           0.           1.        ]]

Rotation (R):
 [[ 0.99680033 -0.05257543 -0.06020729]
 [ 0.04501074  0.99166142 -0.12075452]
 [ 0.06605397  0.11765817  0.9908549 ]]

Translation (T):
 [[-0.17715181]
 [-0.006978  ]
 [ 0.037484  ]]

Saved to hybrid_calibration.npz
```


```bash
Processing 47 valid pairs...
Calibrating Right Camera alone...
Right Camera Initial RMS: 4.88061489293295
Right Camera Matrix Guess:
[[359.06125219   0.         323.54862765]
 [  0.         359.75803205 239.57715766]
 [  0.           0.           1.        ]]
Running Stereo Calibration...

FINAL RMS Error: 4.410858889457142
Translation:
[[-0.12054452]
 [ 0.01117759]
 [-0.17173353]]


```


## For the camera here is the breakdown of results at certain sizing:
```bash
	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 1280x1024
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 1280x960
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 848x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 352x288
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1920x1080
			Interval: Discrete 0.200s (5.000 fps)
			Interval: Discrete 0.333s (3.000 fps)
		Size: Discrete 1280x1024
			Interval: Discrete 0.200s (5.000 fps)
			Interval: Discrete 0.333s (3.000 fps)
		Size: Discrete 1280x960
			Interval: Discrete 0.200s (5.000 fps)
			Interval: Discrete 0.333s (3.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 848x480
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 352x288
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
```




