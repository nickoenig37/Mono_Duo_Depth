import cv2
import pyrealsense2 as rs
import numpy as np

def main():
    # Open both cameras (0 and 1 are common indices; adjust if needed)
    cam_left = cv2.VideoCapture(2)
    cam_right = cv2.VideoCapture(0)

    # Force 640x480 resolution
    WIDTH, HEIGHT = 640, 480
    cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Verify if settings applied
    print("Left cam:", cam_left.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cam_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Right cam:", cam_right.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cam_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if cameras opened successfully
    if not cam_left.isOpened() or not cam_right.isOpened():
        print("Error: One or both cameras could not be opened.")
        return
    
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    while True:
        # Read frames from both cameras
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()
        if not ret_left or not ret_right:
            print("Error: Failed to read from one or both cameras.")
            break

        # Read from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert RealSense frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Resize everything to the same height
        h = 360
        frame_left = cv2.resize(frame_left, (int(frame_left.shape[1] * h / frame_left.shape[0]), h))
        color_image = cv2.resize(color_image, (int(color_image.shape[1] * h / color_image.shape[0]), h))
        frame_right = cv2.resize(frame_right, (int(frame_right.shape[1] * h / frame_right.shape[0]), h))

        # Combine the frames and show the video
        top_row = cv2.hconcat([frame_left, color_image, frame_right])
        depth_colormap_resized = cv2.resize(depth_colormap, (top_row.shape[1], 240))
        combined = cv2.vconcat([top_row, depth_colormap_resized])
        cv2.imshow("Dual Camera + RealSense Feed", combined)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release cameras and close windows
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
