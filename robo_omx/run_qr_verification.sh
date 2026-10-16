#!/usr/bin/env bash
set -e

xhost +local:root >/dev/null 2>&1 || true

echo "=========================================================="
echo " Starting Calibration Verification (Marker Approach)"
echo "=========================================================="
echo "Usage: ./run_qr_verification.sh [MARKER_SIZE_METERS] [HEIGHT_OFFSET_METERS] [MODE]"
echo "Example: ./run_qr_verification.sh 0.05 0.08 auto"
echo "  - MARKER_SIZE: 一辺の長さ (m) [省略=キャリブレーション値を自動使用]"
echo "  - HEIGHT_OFFSET: マーカー上空の停止高さ (m) [デフォルト: 0.08]"
echo "  - MODE: auto / qr / aruco [デフォルト: auto]"
echo "=========================================================="

MARKER_SIZE=${1:-}
HEIGHT_OFFSET=${2:-0.08}
MODE=${3:-auto}
TIP=${4:-"0 0.02 0"}

docker compose run --rm omx-dev /bin/bash -c "
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash 2>/dev/null || true
python3 /root/ros2_ws/workspace/calibration/qr_verification_approach.py \
    ${MARKER_SIZE:+--marker-size ${MARKER_SIZE}} \
    --height-offset ${HEIGHT_OFFSET} \
    --mode ${MODE} \
    --tip-offset ${TIP} \
    --calib-file /root/ros2_ws/workspace/calibration/calibration_result.json
"
