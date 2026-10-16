#!/usr/bin/env bash
# 汎用ランチャー: ./run_task.sh <calibration/内のスクリプト名> [引数...]
# 例: ./run_task.sh example_touch_target.py --table-z 0.0 --height 0.08
set -e
xhost +local:root >/dev/null 2>&1 || true
SCRIPT=${1:?usage: ./run_task.sh script.py [args...]}
shift
docker compose run --rm omx-dev /bin/bash -c "
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash 2>/dev/null || true
cd /root/ros2_ws/workspace/calibration
python3 ${SCRIPT} $*
"
