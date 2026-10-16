#!/usr/bin/env bash
set -e

# RealSense with PointCloud + Calibrated TF Launcher
# Usage: ./run_camera_pointcloud.sh [--no-tf]

PUBLISH_TF=true
if [ "$1" == "--no-tf" ]; then
    PUBLISH_TF=false
fi

xhost +local:root >/dev/null 2>&1 || true

echo "=========================================================="
echo " Starting Intel RealSense with PointCloud (RGB enabled)"
if [ "$PUBLISH_TF" = true ]; then
    echo " + Calibrated Hand-Eye Static TF Publisher"
fi
echo "=========================================================="

if [ "$PUBLISH_TF" = true ]; then
    docker compose run --rm omx-dev /bin/bash -c "
        source /opt/ros/jazzy/setup.bash
        source /root/ros2_ws/install/setup.bash 2>/dev/null || true
        
        # Start calibrated TF publisher in background
        python3 /root/ros2_ws/workspace/calibration/publish_calibrated_tf.py &
        TF_PID=\$!
        
        trap 'kill \$TF_PID 2>/dev/null' EXIT INT TERM
        
        # Launch RealSense with pointcloud enabled
        ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
    "
else
    docker compose run --rm omx-dev /bin/bash -c "
        source /opt/ros/jazzy/setup.bash
        source /root/ros2_ws/install/setup.bash 2>/dev/null || true
        ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
    "
fi
