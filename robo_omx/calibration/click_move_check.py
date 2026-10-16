#!/usr/bin/env python3
"""
click_move_check.py — 動作確認用: 画像をクリックした地点の真上 (既定 2cm) へ手先先端を移動する。

  - 左クリック : その画素の3D位置を求め、真上 --height[m] へ移動
                 (深度が有効なら深度を使用 / 無ければ机面 --table-z を仮定)
  - [H] 初期姿勢へ戻る / [Q] 終了
  - 移動後は一定時間停止して、自動で初期姿勢に戻る (--no-return で無効)
    (初期姿勢からでないと次のクリックの3D位置推定が安定しないため)

安全: 目標 z は --min-z 未満にならないようクランプ。到達不可ならメッセージを出して動かない。
"""
import argparse

import cv2
import numpy as np
import rclpy

from omx_vision import ArmVision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=0.02, help="クリック点の上空[m] (既定 0.02)")
    ap.add_argument("--table-z", type=float, default=0.0, help="深度が無効な時に仮定する机面の world 高さ[m]")
    ap.add_argument("--min-z", type=float, default=0.005, help="手先先端の最低 world 高さ[m] (衝突防止)")
    ap.add_argument("--no-depth", action="store_true", help="深度を使わず机面仮定のみ")
    ap.add_argument("--no-return", action="store_true", help="移動後に初期姿勢へ自動で戻らない")
    ap.add_argument("--dwell", type=float, default=2.0, help="到達後の停止時間[s]")
    args = ap.parse_args()

    rclpy.init()
    arm = ArmVision(use_depth=not args.no_depth)
    arm.wait_ready()
    print("初期姿勢へ移動...")
    arm.move_joints(arm.HOME)

    state = {"click": None, "target": None, "msg": "click a point"}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "click_move_check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    def draw():
        vis = arm.img.copy()
        if state["target"] is not None:
            uv = arm.world_to_pixel(state["target"])
            if uv:
                cv2.drawMarker(vis, (int(uv[0]), int(uv[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, state["msg"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, "click: move above point | H: home | Q: quit", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, vis)
        return cv2.waitKey(1) & 0xFF

    try:
        while rclpy.ok():
            arm.spin_for(0.03)
            k = draw()
            if k in (ord('q'), 27):
                break
            if k in (ord('h'), ord('H')):
                arm.move_joints(arm.HOME)
                continue
            if state["click"] is None:
                continue

            u, v = state["click"]
            state["click"] = None
            if not arm.wait_stationary(3.0):
                state["msg"] = "arm is moving - try again"
                continue

            # 3D位置: 深度 → 机面仮定の順
            src, p = "plane", None
            if not args.no_depth:
                p = arm.pixel_to_world(u, v)
                src = "depth"
            if p is None:
                p = arm.pixel_to_world(u, v, plane_z=args.table_z)
                src = "plane"
            if p is None:
                state["msg"] = "cannot compute 3D point"
                print("3D位置を計算できません")
                continue

            target = p + np.array([0.0, 0.0, args.height])
            if target[2] < args.min_z:
                print(f"目標 z={target[2]:.3f} が min-z={args.min_z} 未満のためクランプ")
                target[2] = args.min_z
            state["target"] = p
            state["msg"] = f"pixel({u},{v}) [{src}] -> {np.round(p, 3)}  target z+{args.height * 100:.0f}cm"
            print(f"pixel=({u},{v}) [{src}] point={np.round(p, 4)} -> tip target={np.round(target, 4)}")
            draw()

            if not arm.move_tip_to(target):
                state["msg"] = "unreachable"
                continue
            tip = arm.tip_world()
            if tip is not None:
                print(f"到達: tip={np.round(tip, 4)}  誤差(mm)={np.round((tip - target) * 1000, 1)} (指令追従)")
            if not args.no_return:
                arm.spin_for(args.dwell)
                arm.move_joints(arm.HOME)
                state["target"] = None
    finally:
        cv2.destroyAllWindows()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
