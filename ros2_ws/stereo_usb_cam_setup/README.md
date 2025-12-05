# Usage Guide 
## Camera's being used:
- 2x [OV2710 USB Camera Modules]
- Lens Specs: 2.8mm, 75 degree FOV



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

## Seeing the channels for all cameras connected to your computer:
This will show you the /video devices connected
```bash
v4l2-ctl --list-devices
```
or for more info on the serial connection you can use:
```bash
for dev in /dev/video*; do echo "$dev"; udevadm info --query=property --name=$dev | grep -E "ID_VENDOR|ID_MODEL|ID_SERIAL"; echo "---"; done
```

## Serial Camera Setup Guide- setting udev rules
**Since the 2 camera's that we have and are setting up are identical models, we need to set up udev rules to distinguish between them based on which USB port they are plugged into. This way, we can always have the left camera and right camera assigned correctly regardless of the order they are plugged in.**

**Whenever you plug in the cameras, make sure to always plug the left camera into the left USB port and the right camera into the right USB port. This is important for maintaining consistent camera assignments.**


## Setting up UDev Rule for distinguishing Cameras.
* Start with only 1 camera plugged in.
**Plug in whatever will be the left camera**
- Then run the command mentioned above for getting information to know which camera is the left one:
```bash
v4l2-ctl --list-devices
```
- Then plug in the second camera (the right one), and do the same thing to know which /video is which camera.

Then run this command where * is the video device number for that camera (e.g. video2):
```bash
udevadm info --attribute-walk --name=/dev/video* | grep "KERNELS" | head -n 1
```
Look for something like this to be outputted:
KERNELS=="1-1.4"

**Get Both Kernels values for both cameras.**

### With this info make a udev rules by port path:
* edit with sudo nano:
```bash
sudo nano /etc/udev/rules.d/99-stereo-cameras.rules
```
ADD:
```bash
# Left Camera (Port 3-5) - Video Stream Only
SUBSYSTEM=="video4linux", KERNELS=="3-5:1.0", ATTR{index}=="0", SYMLINK+="cam_left", MODE="0666"

# Right Camera (Port 3-1) - Video Stream Only
SUBSYSTEM=="video4linux", KERNELS=="3-1:1.0", ATTR{index}=="0", SYMLINK+="cam_right", MODE="0666"
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

## Note about setup for running nodes:
- Based on dependencies, you may need to have setup a venv for running the nodes, I needed to on my laptop- but may not need to on the jetson based on the numpy library.
- If so go to where yours is setup and
```bash
source venv/bin/activate
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