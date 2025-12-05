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
        self.declare_parameter('height', 720)  # Try maximum height for full FOV
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
        
        #    3. CLEANUP: Removed hardcoded brightness/contrast/gamma. 
        # Rely on the camera's internal Auto-Exposure first.
        # 0.75 is often the flag for "Enable Auto Exposure" in V4L2 via OpenCV, but varies by backend.
        # Safest is to NOT set it and let firmware default, or use v4l2-ctl externally.
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"{camera_name} configured: {actual_w}x{actual_h} at {self.frame_rate} FPS")

    def timer_callback(self):
        # Capture timestamp as early as possible
        capture_time = self.get_clock().now().to_msg()
        
        # 1. Grab both (Hardware trigger)
        self.cap_left.grab()
        self.cap_right.grab()

        # 2. Retrieve (Decode)
        ret_left, frame_left = self.cap_left.retrieve()
        ret_right, frame_right = self.cap_right.retrieve()

        # 3. Publish Left
        if ret_left:
            # ERROR WAS HERE: "processed_frame_left = self.process_frame(frame_left)"
            # FIX: Just use frame_left directly
            msg_left = self.bridge.cv2_to_imgmsg(frame_left, encoding='bgr8')
            msg_left.header.stamp = capture_time
            msg_left.header.frame_id = 'camera_left_optical_frame'
            self.left_pub.publish(msg_left)

        # 4. Publish Right
        if ret_right:
            # FIX: Just use frame_right directly
            msg_right = self.bridge.cv2_to_imgmsg(frame_right, encoding='bgr8')
            msg_right.header.stamp = capture_time
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
