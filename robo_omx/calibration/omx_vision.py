#!/usr/bin/env python3
"""
omx_vision.py — OpenMANIPULATOR-X + Eye-in-Hand RealSense 汎用ライブラリ

他のタスクから `from omx_vision import ArmVision` で使い回すための共通部品。

  - キャリブレーション結果 (calibration_result.json) と手先オフセットを自動ロード
  - カメラ画像 → ワールド座標 (ArUco/QR 検出, 任意ピクセル + 地面高さ or 深度)
  - ワールド座標 → ピクセル (対象の画像上へのオーバレイ用)
  - 手先先端(tip)を指定位置へ動かす IK 移動 (停止待ち付き)

座標系: world = ロボットベース (x 前, y 左, z 上)。単位は m / rad。

最小例:
    import rclpy
    from omx_vision import ArmVision
    rclpy.init()
    arm = ArmVision()
    arm.wait_ready()
    arm.move_joints(arm.HOME)
    m = arm.find_marker_world()            # {'pos': np.array([x,y,z]), ...}
    if m: arm.move_tip_to(m['pos'] + [0, 0, 0.08])

設定: 同ディレクトリの robot_config.json (無ければデフォルト)
    {"tip_offset": [0, 0.02, 0], "marker_size": null, "calib_file": null}
"""

import json
import os
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    "tip_offset": [0.0, 0.02, 0.0],   # end_effector_link 座標系での実手先先端オフセット[m] (検証で確定済み)
    "marker_size": None,               # None: calibration_result.json の値
    "calib_file": None,
    "image_topic": "/camera/camera/color/image_raw",
    "info_topic": "/camera/camera/color/camera_info",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",  # align_depth 有効時のみ使用
    "trajectory_topic": "/arm_controller/joint_trajectory",
}


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    path = path or os.path.join(HERE, "robot_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ---------------------------------------------------------
# 運動学
# ---------------------------------------------------------
def _rz(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _ry(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _tr(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


class OmxKinematics:
    """OpenMANIPULATOR-X FK/IK (URDF と一致確認済み: FK-TF 差 0mm)。"""
    LIMITS = [(-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]

    def __init__(self, tip_offset=(0.0, 0.0, 0.0)):
        self.tip_offset = np.array(tip_offset, dtype=float)

    @staticmethod
    def fk(q):
        q1, q2, q3, q4 = q
        return (_tr(0.012, 0, 0) @ _rz(q1) @ _tr(0, 0, 0.0595) @ _ry(q2) @ _tr(0.024, 0, 0.128)
                @ _ry(q3) @ _tr(0.124, 0, 0) @ _ry(q4) @ _tr(0.126, 0, 0))

    def tip(self, q):
        return (self.fk(q) @ np.append(self.tip_offset, 1.0))[:3]

    def ik(self, target, q_init, pitch=1.3, max_iter=500, tol=5e-4):
        """先端(tip)を target[m] に合わせる。pitch は手先ピッチの緩い拘束(rad, 下向き=大)。
        戻り値: (ok, q, 位置誤差[m])"""
        target = np.asarray(target, dtype=float)
        q = np.array(q_init, dtype=float)
        q[0] = np.arctan2(target[1], target[0] - 0.012)
        w, lam = 0.08, 0.01
        for _ in range(max_iter):
            cur = self.tip(q)
            e_pos = target - cur
            if np.linalg.norm(e_pos) < tol:
                break
            e = np.hstack([e_pos, [w * (pitch - q[1:].sum())]])
            f0 = np.hstack([cur, [w * q[1:].sum()]])
            J = np.zeros((4, 4))
            for j in range(4):
                qs = q.copy()
                qs[j] += 1e-6
                J[:, j] = (np.hstack([self.tip(qs), [w * qs[1:].sum()]]) - f0) / 1e-6
            q += 0.5 * J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(4), e)
            for j in range(4):
                q[j] = np.clip(q[j], *self.LIMITS[j])
        err = float(np.linalg.norm(self.tip(q) - target))
        return err < 0.005, q, err


def _quat_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------
class ArmVision(Node):
    JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']
    HOME = [0.0, -0.6, 0.0, 1.97]  # 見下ろし姿勢

    def __init__(self, config_path=None, node_name='omx_vision', use_depth=False):
        super().__init__(node_name)
        self.cfg = load_config(config_path)
        self.kin = OmxKinematics(self.cfg["tip_offset"])
        self.bridge = CvBridge()

        self.K = self.D = None
        self.img = self.img_stamp = None
        self.depth = None
        self.q = list(self.HOME)
        self._q_prev = None
        self._q_time = None
        self._q_speed = 1.0
        self.joints_received = False

        self.T_ee_cam, calib_size = self._load_calib(self.cfg["calib_file"])
        self.marker_size = self.cfg["marker_size"] or calib_size or 0.05

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = (cv2.aruco.DetectorParameters_create()
                             if hasattr(cv2.aruco, 'DetectorParameters_create')
                             else cv2.aruco.DetectorParameters())
        self.qr = cv2.QRCodeDetector()

        self.create_subscription(CameraInfo, self.cfg["info_topic"], self._info_cb, 10)
        self.create_subscription(Image, self.cfg["image_topic"], self._img_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        if use_depth:
            self.create_subscription(Image, self.cfg["depth_topic"], self._depth_cb, 10)
        self._pub = self.create_publisher(JointTrajectory, self.cfg["trajectory_topic"], 10)

    # ---------- 初期化 ----------
    def _load_calib(self, path):
        cands = [path] if path else []
        cands += [os.path.join(HERE, "calibration_result.json"),
                  "/root/ros2_ws/workspace/calibration/calibration_result.json"]
        for c in cands:
            if c and os.path.exists(c):
                with open(c, encoding="utf-8") as f:
                    d = json.load(f)
                self.get_logger().info(f"calibration: {c}")
                return np.array(d["result"]["transform_matrix"]), d.get("meta", {}).get("marker_size_m")
        raise RuntimeError("calibration_result.json が見つかりません")

    def _info_cb(self, m):
        if self.K is None:
            self.K = np.array(m.k).reshape(3, 3)
            self.D = np.array(m.d)

    def _img_cb(self, m):
        self.img = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
        self.img_stamp = m.header.stamp

    def _depth_cb(self, m):
        self.depth = self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough')

    def _js_cb(self, m):
        if not set(self.JOINTS).issubset(m.name):
            return
        q = np.array([m.position[m.name.index(n)] for n in self.JOINTS])
        now = time.time()
        if self._q_prev is not None:
            self._q_speed = float(np.max(np.abs(q - self._q_prev)) / max(now - self._q_time, 1e-3))
        self._q_prev, self._q_time = q, now
        self.q = list(q)
        self.joints_received = True

    # ---------- 待機ユーティリティ ----------
    def spin_for(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_ready(self, timeout=15.0):
        """関節状態・カメラ内部パラメータ・画像が揃うまで待つ。"""
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joints_received and self.K is not None and self.img is not None:
                return True
        raise TimeoutError("joint_states / camera_info / image が受信できません")

    def is_stationary(self):
        return self.joints_received and self._q_speed < 0.02 and (time.time() - self._q_time) < 0.5

    def wait_stationary(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.is_stationary():
                return True
        return False

    # ---------- 動作 ----------
    def move_joints(self, q, duration=3.0, wait=True):
        m = JointTrajectory()
        m.joint_names = self.JOINTS
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        p.time_from_start.sec = int(duration)
        p.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        m.points.append(p)
        self._pub.publish(m)
        if wait:
            self.spin_for(duration + 0.3)
            self.wait_stationary()

    def move_tip_to(self, pos, duration=4.0, pitch=1.3, wait=True):
        """手先先端(tip_offset 込み)を world 座標 pos へ。到達不可なら False。"""
        ok, q, err = self.kin.ik(pos, self.q, pitch=pitch)
        if not ok:
            self.get_logger().warn(f"IK 到達不可: err={err * 1000:.1f}mm target={np.round(pos, 3)}")
            return False
        self.move_joints(q, duration, wait)
        return True

    # ---------- 座標・姿勢 ----------
    def T_world_ee(self):
        """TF のみ使用。取れなければ None。"""
        try:
            tr = self.tf_buffer.lookup_transform('world', 'end_effector_link', rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f"TF unavailable: {e}")
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        T = np.eye(4)
        T[:3, :3] = _quat_R(r.x, r.y, r.z, r.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def tip_world(self):
        T = self.T_world_ee()
        return None if T is None else (T @ np.append(self.kin.tip_offset, 1.0))[:3]

    def T_world_cam(self):
        T = self.T_world_ee()
        return None if T is None else T @ self.T_ee_cam

    def pixel_to_world(self, u, v, plane_z=None, depth_m=None, T_world_cam=None):
        """ピクセル (u,v) を world 座標へ。
        plane_z : その画素が水平面 z=plane_z 上にあると仮定 (深度不要, 地面/机上の物体向け)
        depth_m : カメラ光軸方向の距離 [m] を直接指定 (self.depth があれば自動取得も可)
        """
        T = T_world_cam if T_world_cam is not None else self.T_world_cam()
        if T is None:
            return None
        pts = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.D)[0, 0]
        ray_c = np.array([pts[0], pts[1], 1.0])  # z=1 の光線方向 (カメラ光学系)
        R, o = T[:3, :3], T[:3, 3]
        ray_w = R @ ray_c
        if plane_z is not None:
            if abs(ray_w[2]) < 1e-6:
                return None
            s = (plane_z - o[2]) / ray_w[2]
            return None if s <= 0 else o + s * ray_w
        if depth_m is None:
            depth_m = self.depth_at(u, v)
        if not depth_m:
            return None
        return o + depth_m * ray_w

    def depth_at(self, u, v, win=5):
        """(u,v) 周辺 win x win の有効深度(0除外)の中央値[m]。無効/未受信なら None。
        深度は color に位置合わせ済み (align_depth.enable:=true) の前提。解像度が違えば自動スケール。"""
        if self.depth is None or self.img is None:
            return None
        sx = self.depth.shape[1] / self.img.shape[1]
        sy = self.depth.shape[0] / self.img.shape[0]
        x, y, h = int(u * sx), int(v * sy), win // 2
        patch = self.depth[max(y - h, 0):y + h + 1, max(x - h, 0):x + h + 1].astype(np.float64)
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        d = float(np.median(patch))
        if self.depth.dtype == np.uint16:
            d /= 1000.0  # RealSense 16UC1 は mm
        return d

    def world_to_pixel(self, p_world, T_world_cam=None):
        T = T_world_cam if T_world_cam is not None else self.T_world_cam()
        if T is None:
            return None
        pc = (np.linalg.inv(T) @ np.append(p_world, 1.0))[:3]
        if pc[2] <= 0:
            return None
        uv = self.K @ (pc / pc[2])
        return float(uv[0]), float(uv[1])

    # ---------- マーカー検出 ----------
    def _pose_cam(self, corners, size):
        s = size / 2.0
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T

    def detect_markers(self, img=None, marker_size=None, mode='auto'):
        """画像内の全マーカー。[{'id','type','corners','T_cam'}...]"""
        img = self.img if img is None else img
        size = marker_size or self.marker_size
        out = []
        if mode in ('auto', 'aruco'):
            corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    T = self._pose_cam(c[0].astype(np.float32), size)
                    if T is not None:
                        out.append({'id': int(i), 'type': 'ArUco', 'corners': c[0], 'T_cam': T})
        if mode in ('auto', 'qr') and not out:
            ok, info, pts, _ = self.qr.detectAndDecodeMulti(img)
            if ok and pts is not None:
                for t, p in zip(info, pts):
                    if len(p) == 4:
                        T = self._pose_cam(p.astype(np.float32), size)
                        if T is not None:
                            out.append({'id': t, 'type': 'QR', 'corners': p, 'T_cam': T})
        return out

    def find_marker_world(self, marker_id=None, n=15, timeout=10.0, mode='auto', marker_size=None):
        """停止中に n フレーム集めた中央値でマーカーの world 位置/姿勢を返す。
        戻り値: {'pos','T_world','std_mm','id','type'} または None"""
        poses, last, t0, meta = [], None, time.time(), None
        while len(poses) < n and time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.img is None or self.img_stamp == last:
                continue
            last = self.img_stamp
            if not self.is_stationary():
                poses.clear()
                continue
            T_wc = self.T_world_cam()
            if T_wc is None:
                continue
            for m in self.detect_markers(self.img.copy(), marker_size, mode):
                if marker_id is None or m['id'] == marker_id:
                    poses.append(T_wc @ m['T_cam'])
                    meta = m
                    break
        if len(poses) < max(5, n // 2):
            return None
        P = np.array(poses)
        T = np.eye(4)
        T[:3, 3] = np.median(P[:, :3, 3], axis=0)
        T[:3, :3] = P[len(P) // 2, :3, :3]  # 代表姿勢
        return {'pos': T[:3, 3].copy(), 'T_world': T, 'std_mm': P[:, :3, 3].std(axis=0) * 1000,
                'id': meta['id'], 'type': meta['type']}
