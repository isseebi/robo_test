#!/usr/bin/env python3
"""
qr_consistency_check.py

ハンドアイ・キャリブレーションの整合性検証 (マーカーは固定、アームのみ動かす)。

複数の姿勢でマーカーの world 座標を計測する。マーカーは動いていないので、
全姿勢で同じ値になるはず。ばらつき = キャリブレーション(+FK/PnP)誤差の指標。

使い方:
  python3 qr_consistency_check.py            # 自動: 初期姿勢まわりの姿勢を巡回
  python3 qr_consistency_check.py --manual   # 手動: 姿勢を手で動かし [SPACE] で記録 ([Q] 終了)
マーカーを視野に収められない姿勢は自動的にスキップされる。
"""

import argparse
import json
import time

import cv2
import numpy as np
import rclpy

from qr_verification_approach import Verifier

# 初期姿勢に対する関節角の差分 [d1, d2, d3, d4] (rad)
DELTAS = [
    [0.0, 0.0, 0.0, 0.0],
    [0.25, 0.0, 0.0, 0.0],
    [-0.25, 0.0, 0.0, 0.0],
    [0.0, 0.15, 0.0, 0.0],
    [0.0, -0.15, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, -0.2],
    [0.0, 0.0, 0.15, 0.0],
    [0.0, 0.0, -0.15, 0.0],
    [0.2, 0.1, 0.0, 0.15],
    [-0.2, 0.1, 0.0, -0.15],
    [0.2, -0.1, 0.0, -0.15],
    [-0.2, -0.1, 0.0, 0.15],
]


def settle(node, sec):
    t0 = time.time()
    while time.time() - t0 < sec and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)


def record(node, samples, label):
    m, sd = node.measure()
    if m is None:
        print(f"[{label}] マーカー未検出/動作中/TF不可 → スキップ")
        return
    T = node.tf_world_ee()
    # カメラ-マーカー距離 (参考)
    cam_dist = float(np.linalg.norm((np.linalg.inv(node.T_ee_cam) @ np.linalg.inv(T) @ np.append(m, 1))[:3]))
    samples.append({"label": label, "q": [float(x) for x in node.q], "marker_world": m.tolist(),
                    "std_mm": (sd * 1000).tolist(), "cam_dist_m": cam_dist})
    print(f"[{label}] marker world = {np.round(m, 4)}  dist={cam_dist:.3f}m  frame-std(mm)={np.round(sd * 1000, 1)}")


def report(samples, out):
    if len(samples) < 2:
        print("有効サンプルが2未満のため集計できません")
        return
    P = np.array([s["marker_world"] for s in samples])
    mean = P.mean(axis=0)
    dev = np.linalg.norm(P - mean, axis=1) * 1000
    print("\n" + "=" * 60)
    print(f" 有効姿勢数: {len(samples)}")
    print(f" 平均 marker world : {np.round(mean, 4)} m")
    print(f" 軸別 std (mm)     : x={P[:, 0].std() * 1000:.1f} y={P[:, 1].std() * 1000:.1f} z={P[:, 2].std() * 1000:.1f}")
    print(f" 平均からの距離(mm) : 平均 {dev.mean():.1f} / 最大 {dev.max():.1f}")
    worst = int(dev.argmax())
    print(f" 最大ずれの姿勢    : {samples[worst]['label']} q={np.round(samples[worst]['q'], 2)}")
    # 判断目安
    if dev.max() < 5:
        verdict = "良好 (<5mm)"
    elif dev.max() < 10:
        verdict = "やや大きい (5-10mm): サンプル増やして再キャリブ推奨"
    else:
        verdict = "大きい (>=10mm): 再キャリブ or FK/URDF/マーカーサイズ確認"
    print(f" 判定: {verdict}")
    print("=" * 60)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mean_marker_world": mean.tolist(), "max_dev_mm": float(dev.max()),
                   "mean_dev_mm": float(dev.mean()), "samples": samples}, f, indent=2)
    print(f"結果を保存: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-size", type=float, default=None)
    ap.add_argument("--calib-file", type=str, default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "qr", "aruco"])
    ap.add_argument("--joint2", type=float, default=-0.6)
    ap.add_argument("--joint3", type=float, default=0.0)
    ap.add_argument("--joint4", type=float, default=1.97)
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--move-time", type=float, default=3.0)
    ap.add_argument("--output", default="consistency_result.json")
    args = ap.parse_args()
    args.height_offset = 0.0

    rclpy.init()
    node = Verifier(args)
    samples = []
    try:
        # 起動待ち
        while rclpy.ok() and not (node.joints_received and node.K is not None and node.img is not None):
            rclpy.spin_once(node, timeout_sec=0.1)

        if args.manual:
            cv2.namedWindow("Consistency (manual)", cv2.WINDOW_NORMAL)
            print("姿勢を変えてマーカーを視野に入れ、停止後 [SPACE] で記録、[Q] で終了")
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                vis = node.img.copy()
                corners, _ = node.detect(vis)
                if corners is not None:
                    cv2.polylines(vis, [np.int32(corners)], True, (0, 255, 0), 2)
                cv2.putText(vis, f"samples: {len(samples)}  SPACE=record Q=quit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Consistency (manual)", vis)
                k = cv2.waitKey(10) & 0xFF
                if k == 32:
                    record(node, samples, f"manual{len(samples)}")
                elif k in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()
        else:
            home = np.array([0.0, args.joint2, args.joint3, args.joint4])
            print("初期姿勢へ移動...")
            node.send(home, 4.0)
            settle(node, 5.0)
            for i, d in enumerate(DELTAS):
                q = home + np.array(d)
                node.send(q, args.move_time)
                settle(node, args.move_time + 1.0)
                record(node, samples, f"pose{i}")
            node.send(home, 3.0)
            settle(node, 3.5)
        report(samples, args.output)
    except KeyboardInterrupt:
        report(samples, args.output)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
