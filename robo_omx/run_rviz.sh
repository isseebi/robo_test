#!/usr/bin/env bash
set -e

xhost +local:root >/dev/null 2>&1 || true

echo "Starting RViz2 with RobotModel, TF, and RealSense RGB-PointCloud preset..."
docker compose run --rm omx-dev /bin/bash -c "
    source /opt/ros/jazzy/setup.bash
    source /root/ros2_ws/install/setup.bash 2>/dev/null || true
    rviz2 -d /root/ros2_ws/workspace/open_manipulator_realsense.rviz
"
