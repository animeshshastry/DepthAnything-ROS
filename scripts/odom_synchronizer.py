#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from collections import deque
import numpy as np
# import tf_transformations
from geometry_msgs.msg import Quaternion
import tf2_ros
from geometry_msgs.msg import TransformStamped

class OdomImageSyncNode(Node):
    def __init__(self):
        super().__init__('odom_image_sync')
        
        self.declare_parameter('odom_frame', "odom") 
        self.declare_parameter('base_frame', "base_link") 
        self.declare_parameter('pub_tf', False)  # Whether to publish TF or not

        # Subscribe to odometry and image topics
        self.odom_sub = self.create_subscription(
            Odometry, 
            'odom', 
            self.odom_callback, 
            qos_profile=rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, 
            depth=10
            )
        )
        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        
        # Publisher for synchronized odometry
        self.odom_pub = self.create_publisher(Odometry, 'synced_odom', 10)

        # Buffer to store odometry messages
        self.odom_buffer = deque(maxlen=200)  # Store recent odometry data

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def publish_tf(self, odom_msg):
        """Publish TF from odometry message."""
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def odom_callback(self, msg):
        """Store odometry messages in a buffer."""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.odom_buffer.append((timestamp, msg))

        if self.get_parameter('pub_tf').get_parameter_value().bool_value:
            # Publish TF from odometry
            self.publish_tf(msg)

    def image_callback(self, msg):
        """Find interpolated odometry for each image timestamp."""
        image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if len(self.odom_buffer) < 2:
            self.get_logger().warn("Not enough odometry data for interpolation")
            return

        # Find the two closest odometry timestamps around the image timestamp
        times = [t[0] for t in self.odom_buffer]
        if image_timestamp < times[0] or image_timestamp > times[-1]:
            self.get_logger().warn("Image timestamp is out of odometry buffer range")
            return

        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i + 1]
            if t1 <= image_timestamp <= t2:
                odom1, odom2 = self.odom_buffer[i][1], self.odom_buffer[i + 1][1]
                interp_odom = self.interpolate_odom(odom1, odom2, image_timestamp, t1, t2)
                interp_odom.header.stamp = msg.header.stamp
                interp_odom.header.frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
                interp_odom.child_frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
                interp_odom.pose.covariance = odom2.pose.covariance
                interp_odom.twist.covariance = odom2.twist.covariance
                self.odom_pub.publish(interp_odom)
                return

    def interpolate_odom(self, odom1, odom2, t, t1, t2):
        """Linear interpolation of odometry position and SLERP for orientation."""
        alpha = (t - t1) / (t2 - t1)

        # Interpolate position
        x = (1 - alpha) * odom1.pose.pose.position.x + alpha * odom2.pose.pose.position.x
        y = (1 - alpha) * odom1.pose.pose.position.y + alpha * odom2.pose.pose.position.y
        z = (1 - alpha) * odom1.pose.pose.position.z + alpha * odom2.pose.pose.position.z

        # Interpolate velocity
        vx = (1 - alpha) * odom1.twist.twist.linear.x + alpha * odom2.twist.twist.linear.x
        vy = (1 - alpha) * odom1.twist.twist.linear.y + alpha * odom2.twist.twist.linear.y
        vz = (1 - alpha) * odom1.twist.twist.linear.z + alpha * odom2.twist.twist.linear.z
        wx = (1 - alpha) * odom1.twist.twist.angular.x + alpha * odom2.twist.twist.angular.x
        wy = (1 - alpha) * odom1.twist.twist.angular.y + alpha * odom2.twist.twist.angular.y
        wz = (1 - alpha) * odom1.twist.twist.angular.z + alpha * odom2.twist.twist.angular.z

        # Interpolate orientation using SLERP
        interp_quat = self.slerp(odom1.pose.pose.orientation, odom2.pose.pose.orientation, alpha)

        # Construct interpolated odometry message
        interp_odom = Odometry()
        interp_odom.pose.pose.position.x = x
        interp_odom.pose.pose.position.y = y
        interp_odom.pose.pose.position.z = z
        interp_odom.pose.pose.orientation = interp_quat
        interp_odom.twist.twist.linear.x = vx
        interp_odom.twist.twist.linear.y = vy
        interp_odom.twist.twist.linear.z = vz
        interp_odom.twist.twist.angular.x = wx
        interp_odom.twist.twist.angular.y = wy
        interp_odom.twist.twist.angular.z = wz

        return interp_odom

    def slerp(self, q1, q2, t):
        """
        Perform spherical linear interpolation (SLERP) between two quaternions.
        
        Args:
            q1 (Quaternion): Start orientation.
            q2 (Quaternion): End orientation.
            t (float): Interpolation factor (0 to 1).
        
        Returns:
            Quaternion: Interpolated orientation.
        """
        # Convert ROS quaternions to numpy arrays
        q1_np = np.array([q1.x, q1.y, q1.z, q1.w])
        q2_np = np.array([q2.x, q2.y, q2.z, q2.w])
        
        # Compute the dot product (cosine of the angle)
        dot = np.dot(q1_np, q2_np)

        # If the dot product is negative, negate one quaternion to take the shorter path
        if dot < 0.0:
            q2_np = -q2_np
            dot = -dot

        # Clamp dot product to avoid numerical errors
        dot = np.clip(dot, -1.0, 1.0)

        # Compute interpolation coefficients
        if dot > 0.9995:  # If quaternions are very close, use linear interpolation
            interp_q = (1 - t) * q1_np + t * q2_np
            interp_q /= np.linalg.norm(interp_q)
        else:
            theta_0 = np.arccos(dot)  # Initial angle
            sin_theta_0 = np.sin(theta_0)

            theta = theta_0 * t  # Interpolated angle
            sin_theta = np.sin(theta)

            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0

            interp_q = s1 * q1_np + s2 * q2_np

            self.get_logger().info(f"Large rotation: using slerp")
        # Convert back to ROS Quaternion message
        return Quaternion(x=interp_q[0], y=interp_q[1], z=interp_q[2], w=interp_q[3])

def main(args=None):
    rclpy.init(args=args)
    node = OdomImageSyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
