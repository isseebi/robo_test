#!/usr/bin/env python3
"""
Tブロックをカメラで認識し、上面の重心をタッチしに行く。

キャリブレーション (calibration_result.json) と手先オフセットは omx_vision.ArmVision が自動ロードする。

検出: HSV 色セグメンテーション → 最大輪郭 → 面積/充填率でT字らしさを確認 → 重心(画素)
      → 画素を「ブロック上面 (z = table_z + block_height)」の水平面へレイキャストして world 座標化。
      (深度不要。--depth を付けると深度画像で3D位置を求める)

操作:
  [C]      ブロックを左クリックして色をサンプリング (HSV 範囲を自動設定)
  [SPACE]  検出 → 上空へ移動 → 再計測(視点を変えて精度向上) → 下降してタッチ → 上空へ戻る
  [H]      初期姿勢 / [Q] 終了

例: ./run_task.sh touch_t_block.py --table-z 0.0 --block-height 0.02
    ./run_task.sh touch_t_block.py --hsv 100 120 80 255 50 255   # H_lo H_hi S_lo S_hi V_lo V_hi
"""
import argparse

import cv2
import numpy as np
import rclpy

from omx_vision import ArmVision


def find_block(img, hsv_range, min_area=800):
    """最大の色領域を返す。戻り値: (centroid(u,v), contour, mask, solidity) または None"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    lo, hi = hsv_range
    if lo[0] <= hi[0]:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    else:  # 赤のように色相が折り返す場合
        mask = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), np.array(hi)) | \
            cv2.inRange(hsv, np.array(lo), np.array([179, hi[1], hi[2]]))
    # 反射で白っぽくなった部分(穴/欠け)を埋める: 大きめclose → 外輪郭を塗りつぶし
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 255, cv2.FILLED)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cen = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
    return cen, c, mask, solidity


def sample_hsv(img, u, v, r=6, margin=(12, 70, 70)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    p = hsv[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].reshape(-1, 3)
    h, s, val = np.median(p, axis=0).astype(int)
    if val < 90:  # 黒は色相が不安定なので明度のみで判定
        return [0, 0, 0], [179, 255, min(val + 40, 255)]
    lo = [(h - margin[0]) % 180, max(s - margin[1], 0), max(val - margin[2], 0)]
    hi = [(h + margin[0]) % 180, min(s + margin[1], 255), min(val + margin[2], 255)]
    return lo, hi


def block_world(arm, hsv_range, plane_z, use_depth, n=7):
    """停止中に n フレーム計測した重心 world 位置の中央値。"""
    pts, solids = [], []
    for _ in range(n * 4):
        if len(pts) >= n:
            break
        arm.spin_for(0.05)
        if not arm.is_stationary():
            continue
        r = find_block(arm.img, hsv_range)
        if r is None:
            continue
        cen, _, _, sol = r
        p = arm.pixel_to_world(*cen) if use_depth else arm.pixel_to_world(*cen, plane_z=plane_z)
        if p is not None:
            pts.append(p)
            solids.append(sol)
    if len(pts) < max(3, n // 2):
        return None, None
    return np.median(pts, axis=0), float(np.median(solids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-z", type=float, default=0.0, help="机面の world 高さ[m]")
    ap.add_argument("--block-height", type=float, default=0.02, help="Tブロックの厚み[m] (上面=table-z+この値)")
    ap.add_argument("--hover", type=float, default=0.07, help="アプローチ時の上面からの高さ[m]")
    ap.add_argument("--touch-margin", type=float, default=0.003, help="タッチ位置の上面からの余裕[m] (0で接触)")
    ap.add_argument("--hsv", type=int, nargs=6, metavar=("H_LO", "H_HI", "S_LO", "S_HI", "V_LO", "V_HI"),
                    help="ブロック色の HSV 範囲 (未指定なら黒。[C] で色サンプルも可)")
    ap.add_argument("--black-v", type=int, default=70, help="黒判定の明度V上限(0-255)。反射/暗い机で調整")
    ap.add_argument("--home", type=float, nargs=4, default=[0.0, -0.8, -0.05, 1.97],
                    metavar=("J1", "J2", "J3", "J4"),
                    help="初期撮影姿勢[rad]。既定は従来(0,-0.6,0,1.97)より手先が約3cm高い。"
                         "高いほど広く見えるがカメラは前傾する")
    ap.add_argument("--depth", action="store_true", help="深度画像で3D位置を求める")
    ap.add_argument("--auto", action="store_true", help="起動後、検出できたら自動で1回タッチして終了")
    args = ap.parse_args()

    hsv_range = ([0, 0, 0], [179, 255, args.black_v])  # デフォルト: 黒 (明度Vが低い)
    if args.hsv:
        h0, h1, s0, s1, v0, v1 = args.hsv
        hsv_range = ([h0, s0, v0], [h1, s1, v1])
    plane_z = args.table_z + args.block_height

    rclpy.init()
    arm = ArmVision(use_depth=args.depth)
    arm.wait_ready()
    arm.HOME = list(args.home)
    arm.move_joints(arm.HOME)

    state = {"click": None, "pick": False}

    def on_mouse(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and state["pick"]:
            state["click"] = (x, y)

    cv2.namedWindow("touch_t_block")
    cv2.setMouseCallback("touch_t_block", on_mouse)

    def touch():
        p, sol = block_world(arm, hsv_range, plane_z, args.depth)
        if p is None:
            print("Tブロック未検出 (色範囲/視野を確認)")
            return False
        if not 0.5 < sol < 0.9:
            print(f"警告: 充填率 {sol:.2f} がT字(約0.6-0.8)らしくありません。別物体の可能性")
        print(f"[1/3] 検出 world={np.round(p, 3)} solidity={sol:.2f}")
        top = np.array([p[0], p[1], plane_z])
        if not arm.move_tip_to(top + [0, 0, args.hover]):
            return False
        # 上空から再計測 (視点が変わるので誤差が減る)
        p2, _ = block_world(arm, hsv_range, plane_z, args.depth)
        if p2 is not None:
            print(f"[2/3] 再計測 world={np.round(p2, 3)} 差={np.round((p2 - p) * 1000, 1)}mm")
            top = np.array([p2[0], p2[1], plane_z])
        print("[3/3] タッチ")
        ok = arm.move_tip_to(top + [0, 0, args.touch_margin], duration=3.0)
        arm.spin_for(0.5)
        arm.move_tip_to(top + [0, 0, args.hover], duration=2.0)
        return ok

    print("[C] 色サンプル  [SPACE] タッチ  [H] 初期姿勢  [Q] 終了")
    try:
        while rclpy.ok():
            arm.spin_for(0.03)
            vis = arm.img.copy()
            if hsv_range is not None:
                r = find_block(arm.img, hsv_range)
                if r is not None:
                    cen, c, _, sol = r
                    cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
                    cv2.circle(vis, (int(cen[0]), int(cen[1])), 6, (0, 0, 255), -1)
                    cv2.putText(vis, f"solidity {sol:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    if args.auto:
                        touch()
                        break
            msg = "click block to sample color" if state["pick"] else "C: color | SPACE: touch | H: home | Q: quit"
            cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("touch_t_block", vis)
            k = cv2.waitKey(1) & 0xFF

            if state["click"] is not None:
                u, v = state["click"]
                state.update(click=None, pick=False)
                hsv_range = sample_hsv(arm.img, u, v)
                print(f"HSV range lo={hsv_range[0]} hi={hsv_range[1]}")
            elif k in (ord('c'), ord('C')):
                state["pick"] = True
            elif k == 32:
                if hsv_range is None:
                    print("色範囲が未設定です")
                else:
                    touch()
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
