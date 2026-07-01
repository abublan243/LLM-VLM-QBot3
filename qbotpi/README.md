# QBot3 Pi-Side Nodes

ROS2 Foxy nodes that run on the Raspberry Pi 4 (Ubuntu 20.04) onboard the QBot3.
The host PC GUI joins the same DDS network directly via **rclpy + Cyclone DDS**
— there is no rosbridge layer. Both ends must:

* run the same RMW (`export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`)
* share the same `ROS_DOMAIN_ID`
* be on the same Layer-2 network (or have multicast routed between them)

## Nodes in this directory

| File | Node | Purpose |
|---|---|---|
| [qbot3_base.py](qbot3_base.py) | `qbot3_base` | Hardware interface — IMU, encoders, battery, bumpers, cliff, wheel-drop; subscribes to `/qbot3/cmd_vel` |
| [realsense_node.py](realsense_node.py) | `realsense_node` | RealSense D435i RGB + filtered depth (PNG-compressed) + camera info |
| [motion_controller.py](motion_controller.py) | `motion_controller` | Closed-loop precise distance/turn moves on `/qbot3/precise_cmd` with feedback topics |
| [odometry_node.py](odometry_node.py) | `odometry_node` | Encoder + IMU dead reckoning → `/qbot3/odom` and `odom→base_link` TF (30 Hz) |

See [../CLAUDE.md](../CLAUDE.md) for the full topic surface.

---

## Prerequisites

The Pi must already have ROS2 Foxy installed on Ubuntu 20.04 with the QBot3 hardware drivers (the `qbot3.lib_qbot` Python package shipped by Quanser). If `qbot3_base.py` cannot import that package it falls back to a `MockQBot3` simulator automatically.

### System packages

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-cv-bridge \
    ros-foxy-tf2-ros \
    ros-foxy-tf2-msgs \
    ros-foxy-nav-msgs \
    ros-foxy-sensor-msgs \
    ros-foxy-geometry-msgs \
    ros-foxy-std-msgs \
    ros-foxy-diagnostic-msgs \
    ros-foxy-kobuki-ros-interfaces
```

If `ros-foxy-kobuki-ros-interfaces` is not in your apt index, build it from source:

```bash
mkdir -p ~/qbot3_ws/src && cd ~/qbot3_ws/src
git clone -b foxy-devel https://github.com/kobuki-base/kobuki_ros_interfaces.git
cd ~/qbot3_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select kobuki_ros_interfaces
echo "source ~/qbot3_ws/install/setup.bash" >> ~/.bashrc
```

### Python packages

```bash
python3 -m pip install --user \
    numpy \
    opencv-python \
    pyrealsense2
```

`rclpy` ships with ROS2 Foxy — do not install it from pip.

---

## Install the nodes

Copy this directory onto the Pi (any path works; `~/qbot3_ros/` is fine):

```bash
# From the host PC
scp -r qbotpi/ pi@<pi-ip>:~/qbot3_ros/
```

On the Pi:

```bash
chmod +x ~/qbot3_ros/*.py
```

If you want them on the Python path system-wide and runnable as a colcon package, drop them into `~/qbot3_ws/src/qbot3_pi/qbot3_pi/` and add a minimal `setup.py` with each node registered as an entry point. For demo/grad-project use, running them directly with `python3` is simpler and is what the rest of this guide assumes.

---

## Run order (each in its own terminal)

Source ROS and set the DDS environment in every shell first:

```bash
source /opt/ros/foxy/setup.bash
# if you built kobuki interfaces from source:
source ~/qbot3_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0     # any value 0–232; must match the host
```

**Terminal 1 — base hardware:**
```bash
python3 ~/qbot3_ros/qbot3_base.py
```

**Terminal 2 — RealSense camera:**
```bash
python3 ~/qbot3_ros/realsense_node.py
```

**Terminal 3 — odometry:**
```bash
python3 ~/qbot3_ros/odometry_node.py
```

**Terminal 4 — closed-loop motion controller (optional, only if using `/qbot3/precise_cmd`):**
```bash
python3 ~/qbot3_ros/motion_controller.py
```

There is no rosbridge step — the host's `rclpy` joins the DDS network directly
once it has the same `RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID` and is on the
same subnet.

Verify discovery from the host:
```bash
# On the host (after sourcing /opt/ros/foxy/setup.bash + setting RMW + DOMAIN)
ros2 topic list                 # should see /qbot3/imu, /camera/color/compressed, …
ros2 topic hz /qbot3/imu        # ~50 Hz
```

---

## Quick sanity checks

Confirm topics are publishing:
```bash
ros2 topic list | grep -E '/qbot3|/camera|/motion'
ros2 topic hz /qbot3/imu              # ~50 Hz
ros2 topic hz /qbot3/encoders         # ~50 Hz
ros2 topic hz /qbot3/odom             # ~30 Hz
ros2 topic hz /camera/color/compressed   # ~15 Hz
```

Watch odometry move when you push the robot by hand:
```bash
ros2 topic echo /qbot3/odom --field pose.pose.position
```

Test a precise move (drives 0.3 m forward):
```bash
ros2 topic pub --once /qbot3/precise_cmd geometry_msgs/Twist \
    "{linear: {x: 0.3}, angular: {z: 0.0}}"
ros2 topic echo /motion/feedback   # progress 0–100
ros2 topic echo /motion/result     # final true/false
```

---

## Auto-start on boot (optional)

Create `/etc/systemd/system/qbot3.service`:

```ini
[Unit]
Description=QBot3 ROS2 nodes
After=network-online.target

[Service]
Type=simple
User=ubuntu
Environment="HOME=/home/ubuntu"
Environment="RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
Environment="ROS_DOMAIN_ID=0"
ExecStart=/bin/bash -lc 'source /opt/ros/foxy/setup.bash && \
    python3 /home/ubuntu/qbot3_ros/qbot3_base.py & \
    python3 /home/ubuntu/qbot3_ros/realsense_node.py & \
    python3 /home/ubuntu/qbot3_ros/odometry_node.py & \
    python3 /home/ubuntu/qbot3_ros/motion_controller.py & \
    wait'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qbot3
journalctl -u qbot3 -f
```

---

## Troubleshooting

**`ImportError: No module named qbot3.lib_qbot`** — Quanser SDK is not installed. `qbot3_base.py` will auto-fall-back to `MockQBot3`; sensor topics will still publish synthetic data so the GUI bring-up can be verified.

**`pyrealsense2` import fails on the Pi** — install via `pip install pyrealsense2`. Some Pi images need the librealsense udev rules: `sudo udevadm control --reload-rules && sudo udevadm trigger`.

**`/qbot3/odom` is stationary** — confirm `/qbot3/encoders` is incrementing (`ros2 topic echo /qbot3/encoders`). If ticks change but odom doesn't move, verify `TICKS_PER_METER` matches the constant in `motion_controller.py` (currently 2578).

**Yaw drifts on `/qbot3/odom`** — odometry uses IMU yaw directly, so drift originates in `qbot3_base.py`'s gyro integration. Run `ros2 topic echo /diagnostics` and look at the `Gyroscope` status — if drift > 0.05 rad/s the gyro needs recalibration (keep robot perfectly still on power-up).

**Host can't see Pi topics** — first verify `ros2 topic list` on the Pi shows everything (`/qbot3/imu`, `/camera/...`). Then on the host, in the same shell as `main.py`:
```bash
echo $RMW_IMPLEMENTATION   # must equal the Pi's value
echo $ROS_DOMAIN_ID        # must equal the Pi's value
ros2 daemon stop && ros2 daemon start
ros2 topic list
```
If the host's `ros2 topic list` shows nothing despite matching env vars, it's network/firewall. Confirm both ends are on the same Layer-2 (multicast must reach both NICs). For Wi-Fi APs that block multicast, fall back to FastDDS unicast or run a Cyclone DDS config XML with manual peers.
