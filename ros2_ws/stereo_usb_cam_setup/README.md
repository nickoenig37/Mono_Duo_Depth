# Usage Guide 


Install the following:
```bash
# common packages
sudo apt update
sudo apt install -y python3-opencv python3-pip v4l-utils
# cv_bridge for ROS2 distro (example for Humble)
sudo apt install ros-<distro>-cv-bridge
# if you plan to use pyudev
pip3 install pyudev
```



## Finding USB Serial Information for both cameras
To find the serial number of your USB cameras, you can use the following command:
- Put in video2 for one cam and then video4 for the other
```bash
udevadm info --query=all --name=/dev/video2 | grep ID_SERIAL
```

To actually get important information about the cameras specifically:
```bash
udevadm info -a -n /dev/video2
```

## Setting up UDev Rule for distinguishing Cameras.
* Start with only 1 camera plugged in.
**PUT THE PLUGGED IN CAMERA IN THE LEFT PORT**
- This will be the left camera

Then run:
```bash
udevadm info -q path -n /dev/video2
```
Look for something like 3-1 (that will be the port path for that camera)

I got for what will be the left cam (video2- the first camera I plugged in):
/devices/pci0000:00/0000:00:14.0/usb3/3-5/3-5:1.0/video4linux/video2

Then for the second plugged in camera (the right one):
/devices/pci0000:00/0000:00:14.0/usb3/3-1/3-1:1.0/video4linux/video4

### With this info make a udev rules by port path:
* edit with sudo nano:
```bash
sudo nano /etc/udev/rules.d/99-stereo-cameras.rules
```
ADD:
```bash
# Left camera (plugged into USB3 port 1)
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTRS{busnum}=="3", ATTRS{devpath}=="5", SYMLINK+="cam_left", MODE="0666"

# Right camera (plugged into USB3 port 2)
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTRS{busnum}=="3", ATTRS{devpath}=="1", SYMLINK+="cam_right", MODE="0666"
```


Then reload the udev rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**These symlinks are permanent, as long as each camera stays plugged into its designated USB port**


## Testing which video topics give video output:
Run this script:
```bash
python3 - <<'EOF'
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    ret, frame = (False, None) if ok else (False, None)
    if ok:
        ret, frame = cap.read()
    print(f"/dev/video{i}: {'✅ frame OK' if ret else '⚠️ no frame'}")
    cap.release()
EOF
```

For running the camera these are the possible commands:
```bash
ros2 run stereo_usb_cam_setup dual_camera_node --ros-args -p width:=1280 -p height:=800
```

```bash
ros2 run stereo_usb_cam_setup dual_camera_node --ros-args -p crop_factor:=0.8
```
Also, for checking available resolutions try this:  
```bash
v4l2-ctl --device=/dev/cam_left --set-fmt-video=width=1920,height=1080,pixelformat=MJPG
```