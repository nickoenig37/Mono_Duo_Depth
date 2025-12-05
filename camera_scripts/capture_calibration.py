import cv2
import pyrealsense2 as rs
import numpy as np
import os
import time

def main():
    # --- CONFIGURATION ---
    # Index of your RIGHT OV2710 camera. 
    # Use 'ls /dev/video*' or your 'list_available_cameras.py' to find it.
    # It is likely 0 or 2 depending on which port it's plugged into.
    RIGHT_CAM_INDEX = 4 # Right is camera 8 do 'ls -l /dev/cam*' to confirm
    SAVE_DIR = "calibration_images"
    
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving images to: {os.path.abspath(SAVE_DIR)}")

    # 1. Setup Right USB Camera (OV2710)
    cam_right = cv2.VideoCapture(RIGHT_CAM_INDEX)
    # Set to standard 640x480 for calibration consistency
    cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FORCE DARKER IMAGE
    cam_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = Manual Mode (usually)
    cam_right.set(cv2.CAP_PROP_EXPOSURE, 70)     # Start here. Try 20, 50, 100.
    cam_right.set(cv2.CAP_PROP_BRIGHTNESS, 80)   # Drop brightness significantly
    cam_right.set(cv2.CAP_PROP_CONTRAST, 60)     # Increase contrast slightly

    # Trying to get better contrast:
    
    if not cam_right.isOpened():
        print(f"Error: Could not open Right Camera at index {RIGHT_CAM_INDEX}")
        return

    # 2. Setup RealSense (Left Camera)
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable only RGB stream for calibration (Depth not needed for intrinsics/extrinsics)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense: {e}")
        return

    print("Cameras started. Press 's' to save a pair, 'q' to quit.")
    
    count = 63
    
    try:
        while True:
            # Get frame from USB Camera
            ret, frame_right = cam_right.read()
            if not ret:
                print("Failed to read from Right Camera")
                break

            # Get frame from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert RealSense to Numpy
            frame_left_rs = np.asanyarray(color_frame.get_data())

            # Visualization: Stack them side-by-side
            # Resize just for display if needed, but save the originals!
            display_img = cv2.hconcat([frame_left_rs, frame_right])
            cv2.imshow("Calibration Capture (Left: RS, Right: OV2710)", display_img)

            key = cv2.waitKey(1) & 0xFF
            
            # Save Command
            if key == ord('s'):
                timestamp = int(time.time() * 1000)
                
                # Save Left (RealSense)
                left_filename = os.path.join(SAVE_DIR, f"left_{count:03d}.png")
                cv2.imwrite(left_filename, frame_left_rs)
                
                # Save Right (OV2710)
                right_filename = os.path.join(SAVE_DIR, f"right_{count:03d}.png")
                cv2.imwrite(right_filename, frame_right)
                
                print(f"Saved Pair {count}: {left_filename}, {right_filename}")
                count += 1
                
                # Flash effect on screen
                cv2.imshow("Calibration Capture (Left: RS, Right: OV2710)", 255 - display_img)
                cv2.waitKey(50)

            elif key == ord('q'):
                break

    finally:
        cam_right.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()