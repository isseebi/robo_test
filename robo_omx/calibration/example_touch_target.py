#!/usr/bin/env python3
"""
example_touch_target.py — omx_vision の使用例テンプレート

画像中をクリックした点 (机の上の物体) の真上へ手先を移動する。
  - 左クリック: その画素が机面(--table-z)上にあるとみなし、真上 --height m へ移動
  - [M]: ArUco/QR マーカー直上へ移動
  - [H]: 初期姿勢 / [Q]: 終了
新タスクはこのファイルをコピーして main() 内を書き換えれば良い。
"""
import argparse

import cv2
import numpy as np
import rclpy

from omx_vision import ArmVision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-z", type=float, default=0.0, help="机面の world 高さ[m]")
    ap.add_argument("--height", type=float, default=0.08, help="対象上空の停止高さ[m]")
    ap.add_argument("--depth", action="store_true", help="机面仮定ではなく深度画像で3D位置を求める")
    args = ap.parse_args()

    rclpy.init()
    arm = ArmVision(use_depth=args.depth)
    arm.wait_ready()
    arm.move_joints(arm.HOME)

    click = {"uv": None}
    cv2.namedWindow("omx_vision")
    cv2.setMouseCallback("omx_vision", lambda e, x, y, f, p: click.update(uv=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    try:
        while rclpy.ok():
            arm.spin_for(0.03)
            vis = arm.img.copy()
            cv2.putText(vis, "click: go above point | M: marker | H: home | Q: quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("omx_vision", vis)
            k = cv2.waitKey(1) & 0xFF

            if click["uv"] is not None:
                u, v = click["uv"]
                click["uv"] = None
                if args.depth:
                    p = arm.pixel_to_world(u, v)  # 深度から (無効なら None)
                    if p is None:
                        print("この画素の深度が無効です(反射/近すぎ/範囲外)。別の点を選んでください")
                else:
                    p = arm.pixel_to_world(u, v, plane_z=args.table_z)
                if p is None:
                    pass
                else:
                    print(f"target world = {np.round(p, 3)}")
                    arm.move_tip_to(p + [0, 0, args.height])
            elif k in (ord('m'), ord('M')):
                m = arm.find_marker_world()
                if m is None:
                    print("マーカー未検出")
                else:
                    print(f"marker {m['type']} id={m['id']} world={np.round(m['pos'], 3)}")
                    arm.move_tip_to(m['pos'] + [0, 0, args.height])
            elif k in (ord('h'), ord('H')):
                arm.move_joints(arm.HOME)
            elif k in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
