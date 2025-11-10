import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct # For unpacking the depth header

class DepthDecompressor(Node):
    def __init__(self):
        super().__init__('depth_decompressor')
        self.bridge = CvBridge()

        # Parameters for topic names
        self.declare_parameter('input_topic', '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth')
        self.declare_parameter('output_topic', '/camera/camera/aligned_depth_to_color/image_raw')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.callback,
            10
        )
        self.publisher = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'Subscribing to {input_topic} and publishing to {output_topic}')

    def callback(self, msg: CompressedImage):
        try:
            # Parse the format string
            # Example: "16UC1; compressedDepth"
            depth_fmt_str, compr_type = msg.format.split(';')
            depth_fmt = depth_fmt_str.strip()
            compr_type = compr_type.strip()

            if compr_type != "compressedDepth":
                self.get_logger().error(f"Compression type is not 'compressedDepth'. Received: {compr_type}")
                return

            # Depth images compressed with compressed_depth_image_transport
            # typically have a 12-byte header.
            # This header contains:
            #   - int: compression format (e.g., PNG_COMPRESSION)
            #   - float: depth_quant_a (scale factor for 32FC1)
            #   - float: depth_quant_b (offset for 32FC1)
            depth_header_size = 12 
            
            if len(msg.data) < depth_header_size:
                self.get_logger().error(f"Received compressed depth image data is too short. Expected at least {depth_header_size} bytes, got {len(msg.data)}.")
                return

            raw_data = msg.data[depth_header_size:]

            # Convert byte string to numpy array
            np_arr = np.frombuffer(raw_data, np.uint8)

            # Decode the image. For depth, it's usually PNG.
            # Use cv2.IMREAD_UNCHANGED to preserve bit depth (e.g., 16-bit)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if cv_image is None:
                self.get_logger().error('cv2.imdecode returned None. Image data might be corrupted or header incorrect.')
                return

            # Handle different depth formats
            if depth_fmt == "16UC1":
                # 16UC1 is typically depth in millimeters.
                # No further scaling needed if it's already in 16UC1 after imdecode.
                # Ensure the image type is uint16
                if cv_image.dtype != np.uint16:
                    self.get_logger().warn(f"Decoded 16UC1 image has unexpected dtype: {cv_image.dtype}. Converting to uint16.")
                    cv_image = cv_image.astype(np.uint16)
                encoding_out = '16UC1'

            elif depth_fmt == "32FC1":
                # For 32FC1, the header contains quantization parameters.
                # Reconstruct them to convert to meters.
                # Assuming the header layout: int, float, float
                header_values = struct.unpack('iff', msg.data[:depth_header_size])
                # compression_format = header_values[0] # Not directly used for decoding in this context
                depth_quant_a = header_values[1]
                depth_quant_b = header_values[2]

                # Apply de-quantization
                # This formula is typical for compressed_depth_image_transport for 32FC1
                cv_image_scaled = depth_quant_a / (cv_image.astype(np.float32) - depth_quant_b)
                
                # Set invalid depth values (where cv_image was 0) to 0 in scaled image
                cv_image_scaled[cv_image == 0] = 0

                cv_image = cv_image_scaled
                encoding_out = '32FC1'

            else:
                self.get_logger().error(f"Unsupported depth format: {depth_fmt}")
                return

            # Convert back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding_out)
            ros_image.header = msg.header
            self.publisher.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthDecompressor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()