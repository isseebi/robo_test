#!/usr/bin/env python3
"""
qr_verification_approach.py (rewritten)

ハンドアイ・キャリブレーション検証: マーカー直上へ手先(end_effector_link)を移動する。

旧版からの主な修正点:
  1. マーカーサイズは常に calibration_result.json の値を使用 (--marker-size で明示した場合のみ上書き、
     食い違う場合は警告)。
  2. アーム停止中のフレームのみを使い、複数フレームの中央値で位置を推定 (PnPノイズ低減)。
  3. TF が取れない場合は黙って自前FKへフォールバックせず、計測を拒否する。
  4. 自前FK(omx_fk)とTFの差(URDFとのズレ)を実測して IK 目標を補正する。
  5. 閉ループ補正: 移動後に再計測し、残差(マーカー真上との差)を表示、[R]キーで再アプローチ。
  6. キャリブレーションと同じ cv2.aruco.estimatePoseSingleMarkers 相当(IPPE_SQUARE)で姿勢推定。

操作: [SPACE] アプローチ / [R] 残差を使って補正 / [M] 現在の残差を計測 / [H] 初期姿勢 / [Q] 終了
"""

import argparse
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


# ---------------------------------------------------------
# 運動学
# ---------------------------------------------------------
def rot_z(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def rot_y(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def omx_fk(q):
    """OpenMANIPULATOR-X FK (world -> end_effector_link)。URDFとの差は実行時にTFで補正する。"""
    q1, q2, q3, q4 = q
    return (trans(0.012, 0.0, 0.0) @ rot_z(q1) @ trans(0.0, 0.0, 0.0595)
            @ rot_y(q2) @ trans(0.024, 0.0, 0.128)
            @ rot_y(q3) @ trans(0.124, 0.0, 0.0)
            @ rot_y(q4) @ trans(0.126, 0.0, 0.0))


LIMITS = [(-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]


def tip_pos(q, tip_offset):
    """end_effector_link 座標系で tip_offset だけずれた点のワールド位置。"""
    return (omx_fk(q) @ np.append(tip_offset, 1.0))[:3]


def solve_omx_ik(target_pos, q_init, max_iter=500, tol=5e-4, target_pitch=1.3, tip_offset=(0.0, 0.0, 0.0)):
    """位置優先のDLS IK。位置誤差(m)を返す。"""
    q = np.array(q_init, dtype=float)
    q[0] = np.arctan2(target_pos[1], target_pos[0])
    w = 0.08
    lam = 0.01
    for _ in range(max_iter):
        cur = tip_pos(q, tip_offset)
        e_pos = target_pos - cur
        if np.linalg.norm(e_pos) < tol:
            break
        e = np.hstack([e_pos, [w * (target_pitch - (q[1] + q[2] + q[3]))]])
        J = np.zeros((4, 4))
        f0 = np.hstack([cur, [w * (q[1] + q[2] + q[3])]])
        for j in range(4):
            qs = q.copy()
            qs[j] += 1e-6
            f1 = np.hstack([tip_pos(qs, tip_offset), [w * (qs[1] + qs[2] + qs[3])]])
            J[:, j] = (f1 - f0) / 1e-6
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(4), e)
        q += 0.5 * dq
        for j in range(4):
            q[j] = np.clip(q[j], *LIMITS[j])
    err = np.linalg.norm(tip_pos(q, tip_offset) - target_pos)
    return err < 0.005, q, err


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------
class Verifier(Node):
    N_AVG = 15

    def __init__(self, args):
        super().__init__('qr_verification_approach')
        self.height_offset = args.height_offset
        self.mode = args.mode
        self.tip_offset = np.array(getattr(args, "tip_offset", [0.0, 0.02, 0.0]), dtype=float)
        self.home = [0.0, args.joint2, args.joint3, args.joint4]

        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.img = None
        self.img_stamp = None
        self.q = list(self.home)
        self.q_prev = None
        self.q_speed = 1.0
        self.q_time = None
        self.joints_received = False

        self.T_ee_cam, calib_size = self.load_calibration(args.calib_file)
        if calib_size is None:
            self.marker_size = args.marker_size or 0.05
        elif args.marker_size and abs(args.marker_size - calib_size) > 1e-4:
            self.get_logger().warn(
                f"!!! 指定マーカーサイズ {args.marker_size} m はキャリブレーション時 {calib_size} m と異なります。"
                "奥行きが比例してずれます !!!")
            self.marker_size = args.marker_size
        else:
            self.marker_size = calib_size

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.qr = cv2.QRCodeDetector()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = (cv2.aruco.DetectorParameters_create()
                             if hasattr(cv2.aruco, 'DetectorParameters_create')
                             else cv2.aruco.DetectorParameters())

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.js_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.get_logger().info(f"Marker size: {self.marker_size * 100:.2f} cm / height offset: "
                               f"{self.height_offset * 100:.1f} cm")

    # --- 入出力 ---
    def load_calibration(self, path):
        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            for c in [os.path.join(here, "calibration_result.json"),
                      "/root/ros2_ws/workspace/calibration/calibration_result.json"]:
                if os.path.exists(c):
                    path = c
                    break
        if path is None or not os.path.exists(path):
            raise RuntimeError(f"calibration file not found: {path}")
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        self.get_logger().info(f"Loaded calibration {path} (method={d.get('primary_method')}, "
                               f"samples={d['meta'].get('num_samples')})")
        return np.array(d['result']['transform_matrix']), d['meta'].get('marker_size_m')

    def info_cb(self, m):
        if self.K is None:
            self.K = np.array(m.k).reshape(3, 3)
            self.D = np.array(m.d)

    def img_cb(self, m):
        self.img = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
        self.img_stamp = m.header.stamp

    def js_cb(self, m):
        names = ['joint1', 'joint2', 'joint3', 'joint4']
        if not set(names).issubset(m.name):
            return
        q = [m.position[m.name.index(n)] for n in names]
        now = time.time()
        if self.q_prev is not None and self.q_time is not None:
            self.q_speed = np.max(np.abs(np.array(q) - self.q_prev)) / max(now - self.q_time, 1e-3)
        self.q_prev, self.q_time = np.array(q), now
        self.q = q
        self.joints_received = True

    def is_stationary(self):
        return self.joints_received and self.q_speed < 0.02 and (time.time() - self.q_time) < 0.5

    # --- 検出・姿勢推定 ---
    def detect(self, img):
        if self.mode in ('auto', 'aruco'):
            corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None and len(corners) > 0:
                return corners[0][0].astype(np.float32), 'ArUco'
        if self.mode in ('auto', 'qr'):
            ok, _, pts, _ = self.qr.detectAndDecodeMulti(img)
            if ok and pts is not None and len(pts) > 0 and len(pts[0]) == 4:
                return pts[0].astype(np.float32), 'QR'
        return None, None

    def pose_cam(self, corners):
        s = self.marker_size / 2.0
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None, None
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, rvec, tvec

    def tf_world_ee(self, stamp=None):
        """TFのみ。取得できなければ None (自前FKへ黙ってフォールバックしない)。"""
        try:
            t_query = rclpy.time.Time.from_msg(stamp) if stamp is not None else rclpy.time.Time()
            tr = self.tf_buffer.lookup_transform('world', 'end_effector_link', t_query,
                                                 timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception:
            try:  # スタンプが古い/未来の場合は最新を使用 (停止中のみ呼ぶので影響小)
                tr = self.tf_buffer.lookup_transform('world', 'end_effector_link', rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f"TF world->end_effector_link unavailable: {e}")
                return None
        t, r = tr.transform.translation, tr.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_R(r.x, r.y, r.z, r.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def measure(self, n=None, gui_cb=None):
        """停止中に n フレームを集め、ワールド座標のマーカー位置(中央値)を返す。"""
        n = n or self.N_AVG
        pts, last_stamp = [], None
        t0 = time.time()
        while len(pts) < n and time.time() - t0 < 10.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.img is None or self.K is None or self.img_stamp == last_stamp:
                continue
            last_stamp = self.img_stamp
            if not self.is_stationary():
                pts.clear()
                continue
            corners, _ = self.detect(self.img.copy())
            if corners is None:
                continue
            T_cm, _, _ = self.pose_cam(corners)
            T_we = self.tf_world_ee(self.img_stamp)
            if T_cm is None or T_we is None:
                continue
            pts.append((T_we @ self.T_ee_cam @ T_cm)[:3, 3])
            if gui_cb:
                gui_cb(len(pts), n)
        if len(pts) < max(5, n // 2):
            return None, None
        pts = np.array(pts)
        return np.median(pts, axis=0), pts.std(axis=0)

    # --- 動作 ---
    def send(self, q, dur):
        m = JointTrajectory()
        m.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        p.time_from_start.sec = int(dur)
        p.time_from_start.nanosec = int((dur - int(dur)) * 1e9)
        m.points.append(p)
        self.pub.publish(m)
        self.get_logger().info(f"trajectory: {np.round(q, 3)} ({dur}s)")

    def wait_motion(self, dur):
        t0 = time.time()
        while time.time() - t0 < dur + 1.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def fk_tf_offset(self):
        """TFのee位置 - 自前FKのee位置 (現在関節角で評価)。URDF不一致の検出と補正に使う。"""
        T = self.tf_world_ee()
        if T is None:
            return np.zeros(3)
        return T[:3, 3] - omx_fk(self.q)[:3, 3]

    def approach(self, marker_w, extra=np.zeros(3)):
        target = marker_w.copy() + extra
        target[2] += self.height_offset
        off = self.fk_tf_offset()
        self.get_logger().info(f"FK vs TF offset (m): {np.round(off, 4)}  |off|={np.linalg.norm(off) * 1000:.1f}mm")
        # IK は omx_fk 基準なので、TFとの差を引いて目標を補正
        ok, q, err = solve_omx_ik(target - off, self.q, tip_offset=self.tip_offset)
        if not ok:
            self.get_logger().warn(f"IK 到達不可 (err {err * 1000:.1f}mm)")
            return False
        self.get_logger().info(f">>> target ee: {np.round(target, 4)} m")
        self.send(q, 4.0)
        self.wait_motion(4.0)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-size", type=float, default=None)
    ap.add_argument("--height-offset", type=float, default=0.08)
    ap.add_argument("--calib-file", type=str, default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "qr", "aruco"])
    ap.add_argument("--joint2", type=float, default=-0.6)
    ap.add_argument("--joint3", type=float, default=0.0)
    ap.add_argument("--joint4", type=float, default=1.97)
    ap.add_argument("--tip-offset", type=float, nargs=3, default=[0.0, 0.02, 0.0], metavar=("X", "Y", "Z"),
                    help="end_effector_link座標系での実際の手先先端オフセット[m] (例: 0 0.02 0 = 先端がリンク原点より2cm左)")
    ap.add_argument("--no-initial-move", action="store_true")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = Verifier(args)
    if not args.no_gui:
        cv2.namedWindow("QR Verification View", cv2.WINDOW_NORMAL)

    print("\n[SPACE] アプローチ  [R] 残差補正して再アプローチ  [M] 残差計測  [H] 初期姿勢  [Q] 終了\n")

    marker_w = None      # 最後に測った(アプローチ前)マーカー位置
    residual = np.zeros(3)  # 補正量 (marker_w - 実測ee真下位置)
    last_target = None
    started = False
    t_start = time.time()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not args.no_initial_move and not started and node.joints_received and time.time() - t_start > 1.0:
                node.send(node.home, 3.0)
                started = True
            if node.img is None or node.K is None:
                continue

            vis = node.img.copy()
            corners, mtype = node.detect(vis)
            status = "stationary" if node.is_stationary() else "MOVING"
            if corners is not None:
                cv2.polylines(vis, [np.int32(corners)], True, (0, 255, 0), 2)
                T_cm, rvec, tvec = node.pose_cam(corners)
                T_we = node.tf_world_ee()
                if T_cm is not None and T_we is not None:
                    pw = (T_we @ node.T_ee_cam @ T_cm)[:3, 3]
                    cv2.putText(vis, f"{mtype} cam {np.round(T_cm[:3, 3], 3)}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    cv2.putText(vis, f"world {np.round(pw, 3)} [{status}]", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            else:
                cv2.putText(vis, "no marker", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, "SPACE approach | R refine | M measure | H home | Q quit",
                        (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if args.no_gui:
                key = 255
            else:
                cv2.imshow("QR Verification View", vis)
                key = cv2.waitKey(10) & 0xFF

            if key == 32:
                if not node.is_stationary():
                    print("アーム停止後に再度押してください")
                    continue
                m, sd = node.measure()
                if m is None:
                    print("マーカー計測失敗 (未検出/TF不可/動作中)")
                    continue
                marker_w = m
                residual = np.zeros(3)
                print(f"marker world = {np.round(m, 4)} (std mm {np.round(sd * 1000, 1)})")
                if node.approach(m):
                    last_target = m.copy()
                    last_target[2] += node.height_offset
            elif key in (ord('m'), ord('M')):
                # 現在のee位置 vs 最後のターゲット
                T = node.tf_world_ee()
                if T is None or last_target is None:
                    print("先に SPACE でアプローチしてください")
                    continue
                err = T[:3, 3] - last_target
                print(f"ee actual {np.round(T[:3, 3], 4)} target {np.round(last_target, 4)} "
                      f"error(mm) {np.round(err * 1000, 1)} |e|={np.linalg.norm(err) * 1000:.1f}mm (指令の追従誤差)")
                # 上空のマーカーを今の姿勢から再測定 → 手先がマーカー真上にあるか
                m, sd = node.measure()
                if m is not None and marker_w is not None:
                    d = m - marker_w
                    print(f"再計測マーカー位置 {np.round(m, 4)} / 初回との差(mm) {np.round(d * 1000, 1)}  "
                          f"← 視点依存の誤差(=キャリブレーション誤差の指標)")
                    horiz = (T[:3, 3] - m)[:2]
                    print(f"手先とマーカーの水平差(mm) {np.round(horiz * 1000, 1)}  高さ差 {np.round((T[2, 3] - m[2]) * 1000, 1)} "
                          f"(目標高さ {node.height_offset * 1000:.0f})")
            elif key in (ord('r'), ord('R')):
                if marker_w is None or last_target is None:
                    print("先に SPACE")
                    continue
                m, _ = node.measure()
                if m is None:
                    print("再計測失敗")
                    continue
                T = node.tf_world_ee()
                # 現在の手先(TF)と、再計測したマーカー真上目標の差だけ補正
                want = m.copy()
                want[2] += node.height_offset
                print(f"補正移動量(mm): {np.round((want - T[:3, 3]) * 1000, 1)}")
                node.approach(m)
                last_target = want
            elif key in (ord('h'), ord('H')):
                node.send(node.home, 3.0)
            elif key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
