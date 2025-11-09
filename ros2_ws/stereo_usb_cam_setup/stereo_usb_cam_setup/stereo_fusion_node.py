#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class StereoFusionNode(Node):
    def __init__(self):
        super().__init__('stereo_fusion_node')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_raw', self.left_callback, 10)
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_raw', self.right_callback, 10)

        # Publisher for fused image
        self.fused_pub = self.create_publisher(Image, '/camera/fused/image_raw', 10)

        # Latest frames
        self.left_image = None
        self.right_image = None

        self.get_logger().info("Stereo Fusion Node started, waiting for images...")

    def left_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_fuse()

    def right_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_fuse()

    def try_fuse(self):
        if self.left_image is None or self.right_image is None:
            return

        try:
            fused = self.fuse_images(self.left_image, self.right_image)
            fused_msg = self.bridge.cv2_to_imgmsg(fused, encoding='bgr8')
            fused_msg.header.stamp = self.get_clock().now().to_msg()
            self.fused_pub.publish(fused_msg)
        except Exception as e:
            self.get_logger().error(f"Fusion error: {e}")

    def fuse_images(self, img_left, img_right):
        # Resize both to same size
        h, w = img_left.shape[:2]
        img_right = cv2.resize(img_right, (w, h))

        # Simple overlay with small offset
        offset = 25  # adjust based on camera spacing
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        aligned_left = cv2.warpAffine(img_left, M, (w, h))

        fused = cv2.addWeighted(aligned_left, 0.5, img_right, 0.5, 0)
        return fused



def main(args=None):
    rclpy.init(args=args)
    node = StereoFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
