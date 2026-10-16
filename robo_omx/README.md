# OpenManipulator-X & RealSense ROS 2 (Docker Environment)

既存のDockerイメージ `robotis/open-manipulator:latest` (ROS 2 Jazzy) を使用した、OpenMANIPULATOR-X および Intel RealSense 開発環境です。

---

## 1. ディレクトリ構成

- `src/` : ソースコード用リポジトリ
  - `open_manipulator` : OpenMANIPULATOR-X コントロールパッケージ
  - `DynamixelSDK` : ROBOTIS Dynamixel SDK
  - `dynamixel_hardware_interface` : ROS 2 Control ハードウェアインターフェース
  - `dynamixel_interfaces` : メッセージ型定義
  - `realsense-ros` : Intel RealSense ROS 2 パッケージ
- `docker-compose.yml` : コンテナ設定ファイル (GUI/X11転送、GPU、デバイスアクセス対応)
- `run.sh` : コンテナ起動用スクリプト

---

## 2. コンテナの起動方法

プロジェクトルートディレクトリで以下のスクリプトを実行します。

```bash
./run.sh
```

または `docker compose` コマンドで起動可能です:

```bash
xhost +local:root
docker compose run --rm omx-dev /bin/bash
```

※ コンテナ内では自動で `/opt/ros/jazzy/setup.bash` および `/root/ros2_ws/install/setup.bash` が読み込まれます。

---

## 3. RealSense の起動確認

コンテナ内で以下のコマンドを実行できます:

### (1) カメラ単体の起動 (realsense2_camera)
```bash
ros2 launch realsense2_camera rs_launch.py
```

### (2) 点群 (PointCloud) + キャリブレーションTF付き起動 (推奨)
ホスト側から以下のスクリプトを実行することで、**最初から点群を有効化し、手先とカメラのキャリブレーションTFも同時に配信**できます:
```bash
./run_camera_pointcloud.sh
```
※ TF配信なしでカメラと点群のみ起動したい場合は `./run_camera_pointcloud.sh --no-tf`

### (3) RealSense ツール・ビューワ
```bash
realsense-viewer
```

---

## 4. OpenMANIPULATOR-X の起動確認

### (1) 実機ロボットアームの起動
U2D2 等のシリアルポート（通常 `/dev/ttyUSB0`）が接続されていることを確認し、以下を実行します:

```bash
ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0
```

RViz を同時に起動する場合:
```bash
ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0 start_rviz:=true
```

### (2) シミュレーション (Gazebo)
```bash
ros2 launch open_manipulator_bringup open_manipulator_x_gazebo.launch.py
```

---

## 5. ホスト側での USB シリアル権限設定 (推奨)

ホストマシン上で U2D2 デバイスの遅延タイマーおよびパーミッションを設定するには、ホスト側で以下を実行してください:

```bash
sudo cp src/open_manipulator/open_manipulator_bringup/open-manipulator-cdc.rules /etc/udev/rules.d/99-open-manipulator-cdc.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```
