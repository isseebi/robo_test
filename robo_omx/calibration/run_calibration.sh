#!/usr/bin/env bash
set -e

xhost +local:root >/dev/null 2>&1 || true

echo "=========================================================="
echo " Starting Hand-Eye Calibration for OpenMANIPULATOR-X & RealSense"
echo "=========================================================="
echo "Usage parameters: [MARKER_SIZE_METERS] [MARKER_ID]"
echo "Example: ./calibration/run_calibration.sh 0.05 0"
echo "=========================================================="

MARKER_SIZE=${1:-0.05}
MARKER_ID=${2:-0}

docker compose run --rm omx-dev /bin/bash -c "
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash 2>/dev/null || true
python3 /root/ros2_ws/workspace/calibration/calibrate_hand_eye.py \
    --marker-size ${MARKER_SIZE} \
    --marker-id ${MARKER_ID} \
    --base-frame world \
    --ee-frame end_effector_link \
    --output /root/ros2_ws/workspace/calibration/calibration_result.json
"
