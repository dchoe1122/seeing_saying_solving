import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
"""This script defines a ROS 2 node that subscribes to a camera image topic and saves the latest image to a file at regular intervals.
The node listens to the 'camera/image_raw' topic, converts the incoming ROS image messages to OpenCV format, and saves them as PNG files every 10 seconds. The saved image is named 'latest_image.png'."""
class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            # Defaulted to tb1 topic. TOODO: resolve this in namespace.
            '/tb1/camera/image_raw',  # resolved in namespace
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.last_saved_time = 0.0
        self.save_interval = 10.0  # seconds
        self.output_filename = 'latest_image.png'

    def listener_callback(self, msg):
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        if current_time - self.last_saved_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(self.output_filename, cv_image)
                self.get_logger().info(f"Saved image to {self.output_filename}")
                self.last_saved_time = current_time
            except Exception as e:
                self.get_logger().error(f"Failed to save image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
