#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Tool for OpenMANIPULATOR-X and Intel RealSense.
Uses OpenCV ArUco marker / Charuco board to detect target poses and calculates
the transformation from end_effector_link to camera_color_optical_frame using cv2.calibrateHandEye.
"""

import sys
import os
import json
import math
import argparse
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped


def rotmat_to_euler(R):
    """Calculate roll, pitch, yaw from rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z


def rotmat_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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


def quat_to_rotmat(qx, qy, qz, qw):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])


class HandEyeCalibrator(Node):
    def __init__(self, marker_size=0.05, marker_id=0, base_frame='world', end_effector_frame='end_effector_link', output_file='calibration/calibration_result.json'):
        super().__init__('hand_eye_calibrator')
        self.marker_size = marker_size
        self.target_marker_id = marker_id
        self.base_frame = base_frame
        self.end_effector_frame = end_effector_frame
        self.output_file = output_file

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_cv_image = None

        # ArUco setup (OpenCV 4.6+ / 4.x compatible)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # Calibration data storage: Eye-in-Hand
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []

        self.get_logger().info('Hand-Eye Calibrator node initialized.')
        self.get_logger().info(f'Base frame: {self.base_frame}, End-effector frame: {self.end_effector_frame}')
        self.get_logger().info(f'Tracking ArUco Marker ID: {self.target_marker_id}, Size: {self.marker_size * 100:.1f} cm')
        self.get_logger().info('Controls: Press [SPACE] in image window to capture sample, [C] to calculate, [Q] to quit.')

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Camera intrinsic parameters received.')

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def detect_marker(self, image):
        if self.camera_matrix is None:
            return None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None, None, None

        rvec, tvec = None, None
        for i, mid in enumerate(ids.flatten()):
            if mid == self.target_marker_id:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                break

        return corners, ids, (rvec, tvec)

    def get_robot_pose(self):
        """Get end-effector pose relative to base frame from TF tree."""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                rclpy.time.Time()
            )
            t = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            q = trans.transform.rotation
            R = quat_to_rotmat(q.x, q.y, q.z, q.w)
            return R, t
        except Exception as ex:
            self.get_logger().warn(f'Could not lookup transform {self.base_frame} -> {self.end_effector_frame}: {ex}')
            return None, None

    def capture_sample(self, rvec, tvec):
        R_gripper2base, t_gripper2base = self.get_robot_pose()
        if R_gripper2base is None:
            self.get_logger().error('Failed to get robot end-effector pose from TF!')
            return False

        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = np.array(tvec).reshape(3, 1)

        self.R_gripper2base_list.append(R_gripper2base)
        self.t_gripper2base_list.append(t_gripper2base.reshape(3, 1))
        self.R_target2cam_list.append(R_target2cam)
        self.t_target2cam_list.append(t_target2cam)

        count = len(self.R_gripper2base_list)
        self.get_logger().info(f'Captured sample #{count} successfully!')
        self.get_logger().info(f'  Robot EE pos: {[round(float(x), 4) for x in t_gripper2base]}')
        self.get_logger().info(f'  Camera to marker pos: {[round(float(x), 4) for x in tvec]}')
        return True

    def compute_hand_eye(self):
        num_samples = len(self.R_gripper2base_list)
        if num_samples < 3:
            self.get_logger().error(f'Need at least 3 samples to calculate calibration (currently have {num_samples})!')
            return False

        self.get_logger().info(f'Calculating Eye-in-Hand calibration with {num_samples} samples...')

        methods = {
            'TSAI': cv2.CALIB_HAND_EYE_TSAI,
            'PARK': cv2.CALIB_HAND_EYE_PARK,
            'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
            'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS
        }

        results = {}
        for name, method_flag in methods.items():
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    self.R_gripper2base_list,
                    self.t_gripper2base_list,
                    self.R_target2cam_list,
                    self.t_target2cam_list,
                    method=method_flag
                )
                r, p, y = rotmat_to_euler(R_cam2gripper)
                qx, qy, qz, qw = rotmat_to_quaternion(R_cam2gripper)
                tx, ty, tz = t_cam2gripper.flatten()

                results[name] = {
                    'translation': {'x': float(tx), 'y': float(ty), 'z': float(tz)},
                    'rotation_rpy_deg': {'roll': float(math.degrees(r)), 'pitch': float(math.degrees(p)), 'yaw': float(math.degrees(y))},
                    'rotation_rpy_rad': {'roll': float(r), 'pitch': float(p), 'yaw': float(y)},
                    'quaternion': {'x': float(qx), 'y': float(qy), 'z': float(qz), 'w': float(qw)},
                    'transform_matrix': np.vstack([
                        np.hstack([R_cam2gripper, t_cam2gripper]),
                        [0, 0, 0, 1]
                    ]).tolist()
                }
            except Exception as e:
                self.get_logger().warn(f'Method {name} failed: {e}')

        if not results:
            self.get_logger().error('Calibration failed for all methods.')
            return False

        primary = results.get('TSAI', list(results.values())[0])

        summary = {
            'meta': {
                'num_samples': num_samples,
                'marker_id': self.target_marker_id,
                'marker_size_m': self.marker_size,
                'base_frame': self.base_frame,
                'end_effector_frame': self.end_effector_frame,
                'camera_frame': 'camera_color_optical_frame'
            },
            'primary_method': 'TSAI',
            'result': primary,
            'all_methods': results
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print('\n================ CALIBRATION RESULT (Eye-in-Hand) ================')
        print(f'Saved to: {self.output_file}')
        print(f"Translation [x, y, z] (m): [{primary['translation']['x']:.4f}, {primary['translation']['y']:.4f}, {primary['translation']['z']:.4f}]")
        print(f"Rotation RPY (degrees)   : [r: {primary['rotation_rpy_deg']['roll']:.2f}°, p: {primary['rotation_rpy_deg']['pitch']:.2f}°, y: {primary['rotation_rpy_deg']['yaw']:.2f}°]")
        print(f"Quaternion [x, y, z, w]  : [{primary['quaternion']['x']:.4f}, {primary['quaternion']['y']:.4f}, {primary['quaternion']['z']:.4f}, {primary['quaternion']['w']:.4f}]")
        print('\nROS 2 Static Transform Publisher Command:')
        print(f"ros2 run tf2_ros static_transform_publisher "
              f"--x {primary['translation']['x']:.5f} --y {primary['translation']['y']:.5f} --z {primary['translation']['z']:.5f} "
              f"--qx {primary['quaternion']['x']:.5f} --qy {primary['quaternion']['y']:.5f} --qz {primary['quaternion']['z']:.5f} --qw {primary['quaternion']['w']:.5f} "
              f"--frame-id {self.end_effector_frame} --child-frame-id camera_link")
        print('==================================================================\n')
        return True


def main():
    parser = argparse.ArgumentParser(description='Eye-in-Hand Calibration Tool')
    parser.add_argument('--marker-size', type=float, default=0.05, help='Marker side length in meters (default: 0.05m = 5cm)')
    parser.add_argument('--marker-id', type=int, default=0, help='ArUco marker ID (default: 0, DICT_6X6_250)')
    parser.add_argument('--base-frame', type=str, default='world', help='Robot base frame (default: world)')
    parser.add_argument('--ee-frame', type=str, default='end_effector_link', help='Robot end-effector frame (default: end_effector_link)')
    parser.add_argument('--output', type=str, default='calibration/calibration_result.json', help='Output JSON path')
    args = parser.parse_args()

    rclpy.init()
    node = HandEyeCalibrator(
        marker_size=args.marker_size,
        marker_id=args.marker_id,
        base_frame=args.base_frame,
        end_effector_frame=args.ee_frame,
        output_file=args.output
    )

    cv2.namedWindow('Calibration View', cv2.WINDOW_AUTOSIZE)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)

            if node.latest_cv_image is None:
                continue

            display_img = node.latest_cv_image.copy()
            corners, ids, pose = node.detect_marker(display_img)

            rvec, tvec = pose if pose else (None, None)

            if corners is not None and ids is not None:
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                if rvec is not None and tvec is not None and node.camera_matrix is not None:
                    if hasattr(cv2, 'drawFrameAxes'):
                        cv2.drawFrameAxes(display_img, node.camera_matrix, node.dist_coeffs, rvec, tvec, node.marker_size * 0.75)
                    else:
                        cv2.aruco.drawAxis(display_img, node.camera_matrix, node.dist_coeffs, rvec, tvec, node.marker_size * 0.75)

            samples_count = len(node.R_gripper2base_list)
            status_color = (0, 255, 0) if (rvec is not None) else (0, 0, 255)
            status_text = f"Marker ID {node.target_marker_id}: {'DETECTED' if rvec is not None else 'NOT DETECTED'}"
            cv2.putText(display_img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_img, f"Captured Samples: {samples_count} (Min 3-5 recommended)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, "[SPACE]: Capture  |  [C]: Compute  |  [Q]: Quit", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            cv2.imshow('Calibration View', display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # SPACE: Capture
                if rvec is not None and tvec is not None:
                    node.capture_sample(rvec, tvec)
                else:
                    node.get_logger().warn('Cannot capture: ArUco marker is not currently visible!')
            elif key == ord('c') or key == ord('C'):  # C: Calculate
                node.compute_hand_eye()
            elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC: Quit
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
