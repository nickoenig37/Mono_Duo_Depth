#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DualCameraPublisher(Node):
    def __init__(self):
        super().__init__('dual_camera_publisher')

        # ROS parameters (you can override in launch file)
        self.declare_parameter('left_camera_device', '/dev/cam_left')
        self.declare_parameter('right_camera_device', '/dev/cam_right')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 800)  # Try maximum height for full FOV
        self.declare_parameter('crop_factor', 1.0)  # For FOV adjustment if needed
        self.declare_parameter('use_mjpg', True)  # Set to False to try YUYV format

        left_cam = self.get_parameter('left_camera_device').get_parameter_value().string_value
        right_cam = self.get_parameter('right_camera_device').get_parameter_value().string_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.crop_factor = self.get_parameter('crop_factor').get_parameter_value().double_value
        self.use_mjpg = self.get_parameter('use_mjpg').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.cap_left = cv2.VideoCapture(left_cam)
        self.cap_right = cv2.VideoCapture(right_cam)

        # Set camera properties for maximum resolution
        self.setup_camera_properties(self.cap_left, "left")
        self.setup_camera_properties(self.cap_right, "right")

        # Publishers
        self.left_pub = self.create_publisher(Image, 'camera/left/image_raw', 10)
        self.right_pub = self.create_publisher(Image, 'camera/right/image_raw', 10)

        self.timer = self.create_timer(1.0 / self.frame_rate, self.timer_callback)
        self.get_logger().info("Dual camera publisher node started.")

    def setup_camera_properties(self, cap, camera_name):
        """Set camera properties for maximum resolution and quality"""
        if not cap.isOpened():
            self.get_logger().error(f"Failed to open {camera_name} camera")
            return
        
        # Set buffer size to 1 to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        # Set format - try MJPG or YUYV based on parameter
        if self.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            format_name = "MJPG"
        else:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
            format_name = "YUYV"
        
        # Set camera properties with error handling
        try:
            # Enable auto-exposure and auto-white balance for better color reproduction
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Aperture Priority Mode (full auto)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
            
            # Optimize image quality settings for proper brightness
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)       # Slightly increased brightness
            cap.set(cv2.CAP_PROP_CONTRAST, 45)         # Moderate contrast
            cap.set(cv2.CAP_PROP_SATURATION, 64)       # Default saturation
            cap.set(cv2.CAP_PROP_SHARPNESS, 80)        # Moderate sharpness
            cap.set(cv2.CAP_PROP_GAMMA, 400)           # Default gamma
        except Exception as e:
            self.get_logger().warn(f"Could not set some camera properties for {camera_name}: {e}")
        
        # Verify the settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        buffer_size = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))
        auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        
        self.get_logger().info(f"{camera_name} camera configured:")
        self.get_logger().info(f"  Resolution: {actual_width}x{actual_height}")
        self.get_logger().info(f"  Format: {format_name}")
        self.get_logger().info(f"  FPS: {actual_fps}")
        self.get_logger().info(f"  Buffer size: {buffer_size}")
        self.get_logger().info(f"  Auto exposure: {auto_exposure}")
        self.get_logger().info(f"  Brightness: {brightness}")
        
        if actual_width != self.width or actual_height != self.height:
            self.get_logger().warn(f"{camera_name} camera: Requested {self.width}x{self.height}, got {actual_width}x{actual_height}")

    def process_frame(self, frame):
        """Process frame with optional cropping for FOV matching"""
        if self.crop_factor != 1.0:
            h, w = frame.shape[:2]
            crop_h = int(h / self.crop_factor)
            crop_w = int(w / self.crop_factor)
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            frame = frame[start_h:start_h+crop_h, start_w:start_w+crop_w]
            # Resize back to original resolution
            frame = cv2.resize(frame, (w, h))
        return frame

    def timer_callback(self):
        # Capture timestamp as early as possible for synchronization
        capture_time = self.get_clock().now().to_msg()
        
        # Read from both cameras sequentially but as quickly as possible
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        # Publish both frames with the same timestamp for better synchronization
        if ret_left:
            processed_frame_left = self.process_frame(frame_left)
            msg_left = self.bridge.cv2_to_imgmsg(processed_frame_left, encoding='bgr8')
            msg_left.header.stamp = capture_time
            msg_left.header.frame_id = 'camera_left_optical_frame'
            self.left_pub.publish(msg_left)

        if ret_right:
            processed_frame_right = self.process_frame(frame_right)
            msg_right = self.bridge.cv2_to_imgmsg(processed_frame_right, encoding='bgr8')
            msg_right.header.stamp = capture_time  # Same timestamp for synchronization
            msg_right.header.frame_id = 'camera_right_optical_frame'
            self.right_pub.publish(msg_right)

    def destroy_node(self):
        self.cap_left.release()
        self.cap_right.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
