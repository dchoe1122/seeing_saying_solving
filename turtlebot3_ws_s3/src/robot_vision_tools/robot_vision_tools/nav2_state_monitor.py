# Long term "proper" implementation via FSM. 
# 6/6/2025: So far, started to implement "navgiate_maze" style state machine. 

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from action_msgs.msg import GoalStatusArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import math
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class Nav2StateMonitor(Node):
    def __init__(self):
        super().__init__('nav2_state_monitor')
        # Define custom qos profile for the node
        custom_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.state = 'normal'
        self.last_status = None

        self.list_of_grids = {
            'aisle1_a': {'x': 3.5, 'y': 1.5},
            'aisle1_b': {'x': 5.5, 'y': 1.5},
            'aisle2_a': {'x': 3.5, 'y': -0.5},
            'aisle2_b': {'x': 5.5, 'y': -0.5},
            'aisle3_a': {'x': 3.5, 'y': -2.3},
            'aisle3_b': {'x': 5.5, 'y': -2.3},
            'aisle4_a': {'x': 3.5, 'y': -4.1},
            'aisle4_b': {'x': 5.5, 'y': -4.1},
            'aisle5_a': {'x': 3.5, 'y': -5.9},
            'aisle5_b': {'x': 5.5, 'y': -5.9},
            'aisle6_a': {'x': 3.5, 'y': -7.7},
            'aisle6_b': {'x': 5.5, 'y': -7.7}
        }

        # Publisher for the current robot state
        self.publisher = self.create_publisher(
            String,
            '/tb1/nav2_state',  # resolved under namespace
            custom_qos_profile
        )

        # Publisher for robot velocity (to rotate the robot)
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/tb1/cmd_vel',
            custom_qos_profile)
        
        # Create publisher to publish goal pose
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose',10)     


        # Subscription to goal status
        self.subscription = self.create_subscription(
            GoalStatusArray,
            '/tb1/navigate_to_pose/_action/status',
            self.status_callback,
            10
        )

        # Subscription to /amcl_pose to extract the robot's pose
        # for now, this is hardcoded to be tb1...
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/tb1/amcl_pose',
            self.amcl_pose_callback,
            custom_qos_profile
        )

        # Subscription to camera images
        self.image_subscription = self.create_subscription(
            Image,
            '/tb1/camera/image_raw',  # resolved under namespace
            self.image_callback,
            10
        )

        # Timer to publish the current state periodically
        self.create_timer(1.0, self.publish_state)

        # Image saving attributes
        self.bridge = CvBridge()
        self.last_saved_time = 0.0
        self.save_interval = 10.0  # seconds
        self.output_filename = 'latest_image.png'

        # Goal and robot pose attributes
        self.goal_pose = None
        self.robot_pose = None

    def status_callback(self, msg):
        if not msg.status_list:
            return

        latest_status = msg.status_list[-1].status
        self.last_status = latest_status

        if latest_status == 6:  # ABORTED
            if self.state != 'conflict':
                self.state = 'conflict'
                self.get_logger().warn('State changed to CONFLICT (goal aborted)')
        else:
            if self.state != 'normal':
                self.state = 'normal'
                self.get_logger().info('State changed to NORMAL')

    def goal_callback(self, msg):
        """Callback to store the Nav2 goal pose."""
        self.goal_pose = msg.pose

    def publish_state(self):
        msg = String()
        msg.data = self.state
        self.publisher.publish(msg)

    def rotate_to_goal(self):
        """Rotate the robot to face the goal."""
        if self.goal_pose is None or self.robot_pose is None:
            self.get_logger().warn("Goal or robot pose is not available.")
            return

        # Calculate the angle to the goal
        dx = self.goal_pose.position.x - self.robot_pose.position.x
        dy = self.goal_pose.position.y - self.robot_pose.position.y
        target_angle = math.atan2(dy, dx)

        # Get the current yaw of the robot
        current_yaw = self.get_robot_yaw()

        # Calculate the angular velocity to rotate the robot
        angle_diff = target_angle - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        twist = Twist()
        twist.angular.z = 0.5 if angle_diff > 0 else -0.5

        # Rotate until the robot faces the goal
        while abs(angle_diff) > 0.1:
            self.cmd_vel_publisher.publish(twist)
            rclpy.spin_once(self)
            current_yaw = self.get_robot_yaw()
            angle_diff = target_angle - current_yaw
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

        # Stop the robot after rotation
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def get_robot_yaw(self):
        """Calculate the robot's yaw from its orientation."""
        if self.robot_pose is None:
            return 0.0

        orientation = self.robot_pose.orientation
        _, _, yaw = self.euler_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        return yaw

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """Convert quaternion to Euler angles."""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def image_callback(self, msg):
        if self.state == 'conflict':  # Save image only if the state is 'conflict'
            self.rotate_to_goal()  # Rotate the robot to face the goal
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
    node = Nav2StateMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()