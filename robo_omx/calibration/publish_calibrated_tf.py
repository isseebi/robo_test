#!/usr/bin/env python3
"""
Publish static TF based on Hand-Eye calibration result.
"""
import sys
import json
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import math
import numpy as np

def rotmat_to_quaternion(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw

class StaticTfPublisher(Node):
    def __init__(self, calib_file=None):
        super().__init__('calibrated_camera_tf_publisher')
        self.broadcaster = StaticTransformBroadcaster(self)

        if calib_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate1 = os.path.join(script_dir, "calibration_result.json")
            candidate2 = "calibration/calibration_result.json"
            candidate3 = "/root/ros2_ws/workspace/calibration/calibration_result.json"
            
            for path in [candidate1, candidate2, candidate3]:
                if os.path.exists(path):
                    calib_file = path
                    break
            if calib_file is None:
                calib_file = candidate1
        
        if not os.path.exists(calib_file):
            self.get_logger().error(f"Calibration file {calib_file} not found!")
            sys.exit(1)
            
        with open(calib_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        res = data['result']
        meta = data.get('meta', {})
        ee_frame = meta.get('end_effector_frame', 'end_effector_link')
        now = self.get_clock().now().to_msg()
        
        transforms_to_send = []
        
        # 1. Transform: end_effector_link -> camera_color_optical_frame (Direct calibrated transform)
        t_opt = TransformStamped()
        t_opt.header.stamp = now
        t_opt.header.frame_id = ee_frame
        t_opt.child_frame_id = meta.get('camera_frame', 'camera_color_optical_frame')
        t_opt.transform.translation.x = res['translation']['x']
        t_opt.transform.translation.y = res['translation']['y']
        t_opt.transform.translation.z = res['translation']['z']
        t_opt.transform.rotation.x = res['quaternion']['x']
        t_opt.transform.rotation.y = res['quaternion']['y']
        t_opt.transform.rotation.z = res['quaternion']['z']
        t_opt.transform.rotation.w = res['quaternion']['w']
        transforms_to_send.append(t_opt)
        
        # 2. Transform: end_effector_link -> camera_link (Root of RealSense internal TF tree)
        # RealSense internal:
        # camera_link -> camera_color_frame (dx=0, dy=0.015, dz=0)
        # camera_color_frame -> camera_color_optical_frame (RPY(-pi/2, 0, -pi/2))
        T_ee_copt = np.array(res['transform_matrix'])
        R_ros2opt = np.array([
            [ 0.,  0.,  1.],
            [-1.,  0.,  0.],
            [ 0., -1.,  0.]
        ])
        T_camlink_copt = np.eye(4)
        T_camlink_copt[:3, :3] = R_ros2opt
        T_camlink_copt[:3, 3] = [0.0, 0.015, 0.0]
        
        T_ee_camlink = T_ee_copt @ np.linalg.inv(T_camlink_copt)
        qx, qy, qz, qw = rotmat_to_quaternion(T_ee_camlink[:3, :3])
        
        t_link = TransformStamped()
        t_link.header.stamp = now
        t_link.header.frame_id = ee_frame
        t_link.child_frame_id = 'camera_link'
        t_link.transform.translation.x = float(T_ee_camlink[0, 3])
        t_link.transform.translation.y = float(T_ee_camlink[1, 3])
        t_link.transform.translation.z = float(T_ee_camlink[2, 3])
        t_link.transform.rotation.x = float(qx)
        t_link.transform.rotation.y = float(qy)
        t_link.transform.rotation.z = float(qz)
        t_link.transform.rotation.w = float(qw)
        transforms_to_send.append(t_link)
        
        self.broadcaster.sendTransform(transforms_to_send)
        self.get_logger().info(f"Published static transform: {ee_frame} -> {t_opt.child_frame_id}")
        self.get_logger().info(f"Published static transform: {ee_frame} -> {t_link.child_frame_id}")

def main():
    calib_file = sys.argv[1] if len(sys.argv) > 1 else None
    rclpy.init()
    node = StaticTfPublisher(calib_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
