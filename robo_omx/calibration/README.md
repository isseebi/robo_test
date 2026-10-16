# Eye-in-Hand キャリブレーションツール

ロボット手先付近に固定されたIntel RealSense（Eye-in-Hand構成）とOpenMANIPULATOR-Xの手先リンク（`end_effector_link`）間の相対位置姿勢（変換行列 $T_{ee}^{cam}$）を推定・算出するツールです。

---

## 1. 原理と仕組み

1. **方式**: OpenCV `calibrateHandEye`（TSAI / PARK / HORAUD / DANIILIDIS アルゴリズム）による **Eye-in-Hand** キャリブレーション。
2. **マーカー**: 静止した台・机の上に固定した **ArUcoマーカー**（デフォルト: `DICT_6X6_250`, ID: `0`）。
3. **データ収集**:
   - ロボットの手先を様々な角度・位置（回転を十分に含めるのがコツです）に動かします。
   - カメラ画像からArUcoマーカーの姿勢（$T_{cam}^{marker}$）を推定。
   - ROS 2のTFツリーからベース座標に対する手先リンク姿勢（$T_{base}^{ee}$）を取得。
   - 複数点（推奨5〜10ポーズ）サンプリングして手先〜カメラの変換行列を解きます。

---

## 2. 準備

### (1) ArUcoマーカーの印刷または画面表示
すでに [calibration/marker_0.png](file:///home/isseebi/Desktop/user/robo_omx/calibration/marker_0.png) が生成されています。
- この画像を紙に印刷するか、タブレット/スマホ等の画面に表示し、**机の上など動かない場所に平らに固定**してください。
- マーカーの一辺の長さ（黒枠の外寸）を定規等で測ってください（例: 5cmなら `0.05`m）。

マーカーを再生成する場合:
```bash
docker compose run --rm omx-dev python3 /root/ros2_ws/workspace/calibration/generate_marker.py
```

---

## 3. キャリブレーションの実行手順

### ターミナル 1: ロボットの起動
```bash
./run.sh
# コンテナ内で
ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0
```

### ターミナル 2: RealSenseの起動
別ターミナルでコンテナに入り、カメラを起動します:
```bash
./run.sh
# コンテナ内で
ros2 launch realsense2_camera rs_launch.py
```

### ターミナル 3: キャリブレーションツールの起動
ホスト側から以下のコマンドを実行します:
```bash
# 第1引数: マーカーの一辺の長さ(m) [デフォルト: 0.05 = 5cm]
# 第2引数: マーカーID [デフォルト: 0]
./calibration/run_calibration.sh 0.05 0
```

---

## 4. サンプリング操作

GUIウィンドウ（`Calibration View`）が表示されます。

1. ロボットの手先姿勢を少しずつ変化させ、カメラの視野内にマーカーが映る状態にします。
   - 画面左上に緑色で **`Marker ID 0: DETECTED`** と表示され、マーカー上に3軸が表示されます。
2. キーボードの **`[SPACE]`** キーを押すと、現在のロボット手先姿勢とカメラ認識結果が1サンプル記録されます。
3. ロボットの手先の角度（ピッチやロール、ヨー）を傾けたり、位置をずらしたりして、**5〜10パターン程度サンプリング**します。
   > **高精度化のポイント**: 平行移動だけでなく、手先を傾ける（姿勢回転の変化を大きくする）ことで解の精度が大幅に向上します。
4. 十分なサンプルが取れたら、キーボードの **`[C]`** キーを押します。
   - キャリブレーション計算が実行され、結果がコンソールに表示されるとともに [calibration/calibration_result.json](file:///home/isseebi/Desktop/user/robo_omx/calibration/calibration_result.json) に保存されます。
5. **`[Q]`** キーで終了します。

---

## 5. 結果の適用とTF配信

保存されたキャリブレーション結果を使って、ロボットの手先（`end_effector_link`）からカメラ光学フレーム（`camera_color_optical_frame`）へのStatic TFを配信できます:

```bash
docker compose run --rm omx-dev python3 /root/ros2_ws/workspace/calibration/publish_calibrated_tf.py
```

また、RViz等でロボットモデルとRealSenseのPointCloud/画像が一致しているか確認できます。

---

## 6. 検証ツール

| ツール | 内容 |
|---|---|
| `./run_consistency_check.sh [auto\|manual]` | マーカーを固定し、複数姿勢で world 座標を計測。ばらつき(=キャリブレーション誤差の目安)を出力 |
| `./run_qr_verification.sh "" 0.08 auto ["0 0.02 0"]` | マーカー直上へのアプローチ検証。第1引数を空にするとマーカーサイズは `calibration_result.json` の値を使用。第4引数は手先オフセット(既定 `0 0.02 0`) |

> マーカーサイズは **キャリブレーション時と必ず同じ値** を使うこと(奥行きが比例してずれる)。
> 実機の手先先端と `end_effector_link` 原点のずれは、`robot_config.json` の `tip_offset` で補正する。

---

## 7. 共通ライブラリ `omx_vision.py`

他のタスクでカメラ付きアームを使うための共通部品。キャリブレーション結果と手先オフセットを自動でロードする。

座標系: `world` = ロボットベース(x 前 / y 左 / z 上)、単位は m / rad。

### 設定 `robot_config.json`
```json
{ "tip_offset": [0.0, 0.02, 0.0], "marker_size": null, "calib_file": null }
```
- `tip_offset`: `end_effector_link` 座標系での実際の手先先端オフセット[m]。グリッパー交換時に見直す。
- `marker_size`: `null` なら `calibration_result.json` の値。
- その他 `image_topic` / `info_topic` / `depth_topic` / `trajectory_topic` も上書き可能。

### 主な API (`ArmVision`)
| 機能 | メソッド |
|---|---|
| 起動待ち | `wait_ready()` |
| 関節角で移動 | `move_joints(q, duration, wait)` |
| 手先先端を world 座標へ移動(IK) | `move_tip_to(pos, duration, pitch, wait)` |
| 手先・先端・カメラの姿勢 | `T_world_ee()` / `tip_world()` / `T_world_cam()` |
| 停止判定 | `is_stationary()` / `wait_stationary()` |
| 画像の点 → world | `pixel_to_world(u, v, plane_z=…)` (机面仮定) / `pixel_to_world(u, v)` (深度) |
| world → 画像の点 | `world_to_pixel(p)` |
| マーカー検出 | `detect_markers()` / `find_marker_world(id)` (停止中の中央値) |
| 深度取得 | `depth_at(u, v)` (周辺の中央値, `use_depth=True` 時) |

### 最小例
```python
import rclpy
from omx_vision import ArmVision

rclpy.init()
arm = ArmVision()            # 深度を使う場合 ArmVision(use_depth=True)
arm.wait_ready()
arm.move_joints(arm.HOME)
p = arm.pixel_to_world(320, 240, plane_z=0.0)
arm.move_tip_to(p + [0, 0, 0.08])
```

### 実行方法
```bash
./run_task.sh <calibration/内のスクリプト> [引数...]
./run_task.sh click_move_check.py              # クリック地点の2cm上へ移動 (動作確認用)
./run_task.sh click_move_check.py --no-depth   # 机面仮定のみ
./run_task.sh example_touch_target.py --depth  # テンプレート (深度使用)
```
新しいタスクは `example_touch_target.py` をコピーして作る。

### 深度を使う場合
RealSense を深度位置合わせ付きで起動する(`./run_camera_pointcloud.sh` は設定済み)。
```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
```
深度が無効な画素(反射・近すぎ・範囲外)では `None` が返り、移動しない。

### 注意
- カメラ/グリッパーの取り付けを変えたら再キャリブレーションと `tip_offset` の見直しが必要。
- 深度由来の高さには数 mm〜1 cm の誤差がある。接触に近い動作では余裕をもたせる。
