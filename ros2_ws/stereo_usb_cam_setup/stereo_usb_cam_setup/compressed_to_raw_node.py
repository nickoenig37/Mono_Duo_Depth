import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class CompressedImageRepublisher(Node):
    def __init__(self):
        super().__init__('compressed_to_raw_republisher')

        # Parameters (can be set from launch file)
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('output_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_encoding', 'bgr8')  # or 'mono8' or '16UC1' for depth

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.encoding = self.get_parameter('image_encoding').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.callback,
            10
        )

        self.get_logger().info(f'Republishing {input_topic} -> {output_topic} as {self.encoding}')

    def callback(self, msg):
        try:
            # Convert raw bytes to NumPy array
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if cv_image is None:
                self.get_logger().error("cv2.imdecode returned None. Possibly corrupted image.")
                return

            # Sanity check: if encoding is 16UC1, ensure correct dtype
            if self.encoding == '16UC1' and cv_image.dtype != np.uint16:
                self.get_logger().warn("Expected 16UC1 but image is not uint16. Forcing conversion.")
                cv_image = cv_image.astype(np.uint16)

            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding=self.encoding)
            ros_image.header = msg.header
            self.publisher.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
