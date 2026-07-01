#!/bin/bash

echo "=========================================="
echo " Starting QBot3 Desktop Application"
echo "=========================================="

echo "[1/4] Sourcing ROS 2 Jazzy..."
source /opt/ros/jazzy/setup.bash

echo "[2/4] Sourcing local workspace..."
source /home/waleed/ros2_ws/install/setup.bash

echo "[3/4] Configuring CycloneDDS (Domain ID: 0)..."
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0

echo "[4/4] Activating Python virtual environment..."
source /home/waleed/python-env/bin/activate

echo "Launching application..."
cd /home/waleed/Desktop/VLA_Qbot
python3 main.py
