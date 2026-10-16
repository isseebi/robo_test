#!/usr/bin/env bash
set -e
xhost +local:root >/dev/null 2>&1 || true
# 使い方: ./run_consistency_check.sh [auto|manual] [MODE: auto/qr/aruco]
# マーカーサイズは calibration_result.json の値を自動使用
RUN=${1:-auto}
MODE=${2:-auto}
EXTRA=""
[ "$RUN" = "manual" ] && EXTRA="--manual"
docker compose run --rm omx-dev /bin/bash -c "
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash 2>/dev/null || true
cd /root/ros2_ws/workspace/calibration
python3 qr_consistency_check.py --mode ${MODE} ${EXTRA} --output /root/ros2_ws/workspace/calibration/consistency_result.json
"
