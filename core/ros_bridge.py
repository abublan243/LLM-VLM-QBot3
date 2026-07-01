"""
ROS Bridge — Native DDS (rclpy + Cyclone DDS) connection to the QBot3 Pi.

Replaces the previous rosbridge_websocket / roslibpy implementation. The host
PC now joins the same DDS network as the Pi: matching ``ROS_DOMAIN_ID`` and
``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` make pub/sub Just Work, with no
WebSocket layer, no base64 round-trips, no CBOR overhead.

Threading model
---------------
``rclpy`` callbacks fire on a daemon spin thread. Each callback writes to the
shared state and emits a Qt signal. Qt's queued auto-connection delivers the
signal to slots on the GUI thread safely.

Public surface (preserved from the old WebSocket bridge)
--------------------------------------------------------
* Class name: ``ROSBridge(QObject)``
* Same Qt signals: ``connection_changed``, ``rgb_updated``, ``depth_updated``,
  ``depth_visual_updated``, ``camera_info_updated``, ``imu_updated``,
  ``encoders_updated``, ``odom_updated``, ``battery_updated``,
  ``bump_event``, ``cliff_event``, ``wheel_drop_event``, ``motion_feedback``,
  ``motion_result``, ``motion_status``
* Same publishers: ``publish_cmd_vel``, ``publish_precise_cmd``,
  ``publish_emergency_stop``
* Same constructor signature is accepted; legacy ``host`` / ``port`` kwargs
  are tolerated and ignored so existing call sites don't break.

Requirements
------------
* ROS2 (Foxy on Ubuntu 20.04 to match the Pi, or any compatible distro on
  the same DDS domain)
* ``cv_bridge`` package (ships with ROS2; ``apt install ros-<distro>-cv-bridge``)
* ``rclpy`` (also from the ROS2 install; not a pip package)
* Set environment variables before launching ``main.py``:
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      export ROS_DOMAIN_ID=<same as Pi>
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.shared_state import (
    BatteryReading,
    CameraIntrinsics,
    IMUReading,
    OdometryReading,
    SharedState,
)

logger = logging.getLogger(__name__)


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# =====================================================================
# ROSBridge QObject — Native DDS implementation
# =====================================================================


class ROSBridge(QObject):
    """rclpy-driven bridge that joins the Pi's DDS network directly."""

    # ---- Connection ----
    connection_changed = pyqtSignal(bool, str)        # (connected, message)

    # ---- Sensor / vision update notifiers (no payload — read SharedState) ----
    rgb_updated = pyqtSignal()
    depth_updated = pyqtSignal()
    depth_visual_updated = pyqtSignal()
    camera_info_updated = pyqtSignal()
    imu_updated = pyqtSignal()
    encoders_updated = pyqtSignal()
    odom_updated = pyqtSignal()
    battery_updated = pyqtSignal()

    # ---- Discrete safety events ----
    bump_event = pyqtSignal(int, bool)                # sensor_idx, active
    cliff_event = pyqtSignal(int, bool)
    wheel_drop_event = pyqtSignal(int, bool)

    # ---- Closed-loop motion controller feedback ----
    motion_feedback = pyqtSignal(float)               # progress 0–100
    motion_result = pyqtSignal(bool)
    motion_status = pyqtSignal(str)

    # ---- Pi-side gyro calibration status ----
    imu_calibration_changed = pyqtSignal(bool, float)  # (calibrated, progress 0–1)

    # =================================================================
    def __init__(
        self,
        topics: Dict[str, str],
        state: Optional[SharedState] = None,
        *,
        domain_id: Optional[int] = None,
        rmw_implementation: str = "rmw_cyclonedds_cpp",
        node_name: str = "qbot3_host",
        connection_check_hz: float = 1.0,
        parent: Optional[QObject] = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self._topics = topics
        self.state = state or SharedState.instance()

        self._domain_id = domain_id
        self._rmw = rmw_implementation
        self._node_name = node_name

        # Quietly accept (and discard) the old WebSocket kwargs
        for legacy in ("host", "port", "reconnect_initial_delay", "reconnect_max_delay"):
            legacy_kwargs.pop(legacy, None)
        if legacy_kwargs:
            logger.debug("ROSBridge: ignoring legacy kwargs %s", list(legacy_kwargs))

        # rclpy state — populated in start()
        self._rclpy: Any = None
        self._node: Any = None
        self._cv_bridge: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._publishers: Dict[str, Any] = {}
        self._subscribers: List[Any] = []

        self._stopping = False
        self._connected_emitted = False

        # Connection-status poll
        self._conn_timer = QTimer(self)
        self._conn_timer.setInterval(int(1000 / max(0.2, connection_check_hz)))
        self._conn_timer.timeout.connect(self._poll_connection)

    # ---------------------------------------------------------------
    # Connection lifecycle
    # ---------------------------------------------------------------

    def start(self) -> None:
        """Initialise rclpy, create the node, spawn the spin thread."""
        if self._node is not None:
            return
        if self._rmw and not os.environ.get("RMW_IMPLEMENTATION"):
            os.environ["RMW_IMPLEMENTATION"] = self._rmw
        if self._domain_id is not None and not os.environ.get("ROS_DOMAIN_ID"):
            os.environ["ROS_DOMAIN_ID"] = str(self._domain_id)

        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from cv_bridge import CvBridge
        except Exception as exc:
            logger.error(
                "rclpy / cv_bridge import failed: %s — install ROS2 + cv_bridge "
                "and source the setup.bash before launching main.py.", exc,
            )
            self.state.set_connection(False, "rclpy not installed")
            self.connection_changed.emit(False, "rclpy not installed")
            return

        try:
            if not rclpy.ok():
                rclpy.init()
        except Exception as exc:
            logger.exception("rclpy.init failed: %s", exc)
            self.state.set_connection(False, f"rclpy init failed: {exc}")
            self.connection_changed.emit(False, f"rclpy init failed: {exc}")
            return

        self._rclpy = rclpy
        self._cv_bridge = CvBridge()
        self._node = rclpy.create_node(self._node_name)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        try:
            self._setup_subscriptions(sensor_qos, reliable_qos)
            self._setup_publishers(reliable_qos)
        except Exception as exc:
            logger.exception("Subscription / publisher setup failed: %s", exc)
            self.state.set_connection(False, f"setup failed: {exc}")
            self.connection_changed.emit(False, f"setup failed: {exc}")
            return

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._stopping = False
        self._connected_emitted = False

        self._spin_thread = threading.Thread(
            target=self._spin_loop, name="rclpy-spin", daemon=True,
        )
        self._spin_thread.start()
        self._conn_timer.start()

        domain = os.environ.get("ROS_DOMAIN_ID", "0")
        rmw = os.environ.get("RMW_IMPLEMENTATION", "(default)")
        logger.info(
            "ROSBridge online — node=%s, domain=%s, rmw=%s",
            self._node_name, domain, rmw,
        )

    def stop(self) -> None:
        """Stop the spin thread, destroy the node, shutdown rclpy."""
        self._stopping = True
        self._conn_timer.stop()
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if self._rclpy is not None:
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._spin_thread = None
        self.state.set_connection(False, "stopped")
        self.connection_changed.emit(False, "stopped")

    @property
    def is_connected(self) -> bool:
        return self.state.ros_connected

    # ---- internal ---------------------------------------------------

    def _spin_loop(self) -> None:
        """Run rclpy's executor until shutdown. Daemon thread."""
        try:
            while (
                self._rclpy is not None
                and self._rclpy.ok()
                and not self._stopping
                and self._executor is not None
            ):
                # 100 ms slices so we can check _stopping and exit promptly
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            logger.exception("rclpy spin loop crashed: %s", exc)

    def _poll_connection(self) -> None:
        """Discover whether the Pi is publishing on the IMU topic."""
        if self._node is None:
            return
        topic = self._topics.get("imu", "/qbot3/imu")
        try:
            n_pub = int(self._node.count_publishers(topic))
        except Exception:
            n_pub = 0
        connected = n_pub > 0
        was_connected = self.state.ros_connected
        if connected != was_connected:
            message = "DDS peers discovered" if connected else "no DDS peers"
            self.state.set_connection(connected, message)
            self.connection_changed.emit(connected, message)
        elif connected and not self._connected_emitted:
            self.state.set_connection(True, "DDS peers discovered")
            self.connection_changed.emit(True, "DDS peers discovered")
            self._connected_emitted = True

    # ---------------------------------------------------------------
    # Topic wiring
    # ---------------------------------------------------------------

    def _setup_subscriptions(self, sensor_qos: Any, reliable_qos: Any) -> None:
        from sensor_msgs.msg import BatteryState, CameraInfo, CompressedImage, Imu
        from std_msgs.msg import Bool, Float32, Int64MultiArray, String
        from nav_msgs.msg import Odometry

        # kobuki_ros_interfaces — must be installed on the host as well
        try:
            from kobuki_ros_interfaces.msg import (
                BumperEvent,
                CliffEvent,
                WheelDropEvent,
            )
            kobuki_ok = True
        except ImportError as exc:
            logger.warning(
                "kobuki_ros_interfaces not found — bumper/cliff/wheel_drop "
                "topics will be skipped. Build it from source on the host: %s", exc,
            )
            kobuki_ok = False
            BumperEvent = CliffEvent = WheelDropEvent = None  # type: ignore[assignment]

        T = self._topics
        node = self._node
        assert node is not None

        def sub(name: str, msg_type: Any, callback: Callable[[Any], None],
                qos: Any = sensor_qos) -> None:
            topic_name = T.get(name)
            if not topic_name or msg_type is None:
                return
            handle = node.create_subscription(msg_type, topic_name, callback, qos)
            self._subscribers.append(handle)
            logger.debug("Subscribed %s (%s)", topic_name, msg_type.__name__)

        # Camera (best-effort, lossy is fine for vision streams)
        sub("camera_color", CompressedImage, self._on_color_image)
        sub("camera_depth_raw", CompressedImage, self._on_depth_image)
        sub("camera_depth_visual", CompressedImage, self._on_depth_visual)
        sub("camera_info", CameraInfo, self._on_camera_info, qos=reliable_qos)

        # Robot sensors
        sub("imu", Imu, self._on_imu)
        sub("encoders", Int64MultiArray, self._on_encoders, qos=reliable_qos)
        sub("odom", Odometry, self._on_odom)
        sub("battery", BatteryState, self._on_battery, qos=reliable_qos)

        # Safety events
        if kobuki_ok:
            sub("bumpers", BumperEvent, self._on_bumper, qos=reliable_qos)
            sub("cliff", CliffEvent, self._on_cliff, qos=reliable_qos)
            sub("wheel_drop", WheelDropEvent, self._on_wheel_drop, qos=reliable_qos)

        # Motion controller feedback
        sub("motion_feedback", Float32, self._on_motion_feedback, qos=reliable_qos)
        sub("motion_result", Bool, self._on_motion_result, qos=reliable_qos)
        sub("motion_status", String, self._on_motion_status, qos=reliable_qos)

        # Gyro calibration status (5 s startup + online refinement)
        sub("imu_calibrated", Bool, self._on_imu_calibrated, qos=reliable_qos)
        sub("imu_calibration_progress", Float32, self._on_imu_calibration_progress, qos=reliable_qos)

    def _setup_publishers(self, reliable_qos: Any) -> None:
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Bool

        T = self._topics
        node = self._node
        assert node is not None

        for key in ("cmd_vel", "precise_cmd"):
            name = T.get(key)
            if not name:
                continue
            self._publishers[key] = node.create_publisher(Twist, name, reliable_qos)

        # Optional remote-zero topic for the Pi-side odometry node.
        odom_reset_name = T.get("odom_reset")
        if odom_reset_name:
            self._publishers["odom_reset"] = node.create_publisher(
                Bool, odom_reset_name, reliable_qos,
            )
        # Optional remote yaw-reset / gyro recalibration trigger.
        imu_reset_name = T.get("imu_reset_yaw")
        if imu_reset_name:
            self._publishers["imu_reset_yaw"] = node.create_publisher(
                Bool, imu_reset_name, reliable_qos,
            )
        # Skip calibration — operator clicked the Skip button during the
        # 5 s window to abort it early.
        imu_skip_name = T.get("imu_skip_calibration")
        if imu_skip_name:
            self._publishers["imu_skip_calibration"] = node.create_publisher(
                Bool, imu_skip_name, reliable_qos,
            )

    # ---------------------------------------------------------------
    # Subscription callbacks  (run on rclpy spin thread)
    # ---------------------------------------------------------------

    def _on_color_image(self, msg: Any) -> None:
        try:
            frame = self._cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                return
            self.state.set_rgb_frame(frame)
            self.rgb_updated.emit()
        except Exception as exc:
            logger.exception("Color image decode failed: %s", exc)

    def _on_depth_image(self, msg: Any) -> None:
        try:
            # depth is PNG-encoded uint16 inside CompressedImage; cv_bridge
            # dispatches to cv2.imdecode(IMREAD_UNCHANGED) for non-jpeg formats
            depth = self._cv_bridge.compressed_imgmsg_to_cv2(msg)
            if depth is None:
                return
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            # Fill speckle holes so SLAM / 3D viewer / YOLO distance lookups
            # all see a clean depth map instead of red specks.
            from core.sensor_processor import fill_depth_holes
            depth = fill_depth_holes(depth)
            self.state.set_depth_frame(depth)
            self.depth_updated.emit()
        except Exception as exc:
            logger.exception("Depth image decode failed: %s", exc)

    def _on_depth_visual(self, msg: Any) -> None:
        try:
            frame = self._cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                return
            self.state.set_depth_visual(frame)
            self.depth_visual_updated.emit()
        except Exception as exc:
            logger.exception("Depth visual decode failed: %s", exc)

    def _on_camera_info(self, msg: Any) -> None:
        try:
            k = list(msg.k) if hasattr(msg, "k") else []
            if len(k) < 9:
                return
            intr = CameraIntrinsics(
                fx=float(k[0]), cx=float(k[2]),
                fy=float(k[4]), cy=float(k[5]),
                width=int(getattr(msg, "width", 640)),
                height=int(getattr(msg, "height", 480)),
            )
            self.state.set_camera_intrinsics(intr)
            self.camera_info_updated.emit()
        except Exception as exc:
            logger.exception("CameraInfo parse failed: %s", exc)

    def _on_imu(self, msg: Any) -> None:
        try:
            o = msg.orientation
            qx, qy, qz, qw = float(o.x), float(o.y), float(o.z), float(o.w)
            ang = msg.angular_velocity
            acc = msg.linear_acceleration
            reading = IMUReading(
                quat=(qx, qy, qz, qw),
                yaw_rad=_quat_to_yaw(qx, qy, qz, qw),
                angular_velocity=(float(ang.x), float(ang.y), float(ang.z)),
                linear_acceleration=(float(acc.x), float(acc.y), float(acc.z)),
                monotonic_ts=time.monotonic(),
            )
            self.state.set_imu(reading)
            self.imu_updated.emit()
        except Exception as exc:
            logger.exception("IMU parse failed: %s", exc)

    def _on_encoders(self, msg: Any) -> None:
        try:
            data = list(msg.data) if hasattr(msg, "data") else []
            if len(data) >= 2:
                self.state.set_encoders(int(data[0]), int(data[1]))
                self.encoders_updated.emit()
        except Exception as exc:
            logger.exception("Encoders parse failed: %s", exc)

    # Defensive thresholds — protect the SLAM grid from a single bogus
    # odom message. The user has hit a 200 km pose excursion in the wild
    # caused by a broken encoder rolling over; ABS_LIMIT keeps that from
    # ever pollluting the host-side state, JUMP_LIMIT_M catches smaller
    # but still-impossible per-message jumps. Both are loose enough that
    # legitimate fast motion is fine.
    _ODOM_ABS_LIMIT_M = 10_000.0
    _ODOM_JUMP_LIMIT_M = 5.0

    def _on_odom(self, msg: Any) -> None:
        try:
            pose = msg.pose.pose
            twist = msg.twist.twist
            new_x = float(pose.position.x)
            new_y = float(pose.position.y)

            # Hard absolute sanity check — no real run will see |x|>10 km
            if abs(new_x) > self._ODOM_ABS_LIMIT_M or abs(new_y) > self._ODOM_ABS_LIMIT_M:
                logger.warning(
                    "Rejecting bogus odom: x=%.1f y=%.1f (>%g m absolute limit)",
                    new_x, new_y, self._ODOM_ABS_LIMIT_M,
                )
                return

            # Per-message jump check — if the previous reading was valid
            # and the new one is more than JUMP_LIMIT_M away, drop it.
            with self.state.lock:
                last = self.state.odom
            if last.monotonic_ts > 0:
                dx = new_x - last.x
                dy = new_y - last.y
                if dx * dx + dy * dy > self._ODOM_JUMP_LIMIT_M ** 2:
                    logger.warning(
                        "Rejecting odom jump: Δ=(%.2f, %.2f) m  (limit %.1f m)",
                        dx, dy, self._ODOM_JUMP_LIMIT_M,
                    )
                    return

            yaw = _quat_to_yaw(
                float(pose.orientation.x), float(pose.orientation.y),
                float(pose.orientation.z), float(pose.orientation.w),
            )
            reading = OdometryReading(
                x=new_x,
                y=new_y,
                yaw_rad=yaw,
                linear_x=float(twist.linear.x),
                angular_z=float(twist.angular.z),
                monotonic_ts=time.monotonic(),
            )
            self.state.set_odom(reading)
            self.odom_updated.emit()
        except Exception as exc:
            logger.exception("Odom parse failed: %s", exc)

    def _on_battery(self, msg: Any) -> None:
        try:
            voltage = float(msg.voltage)
            v_full = 12.6
            v_empty = 10.5
            percent = max(0.0, min(100.0, (voltage - v_empty) / (v_full - v_empty) * 100.0))
            self.state.set_battery(BatteryReading(
                voltage=voltage, percent=percent, monotonic_ts=time.monotonic(),
            ))
            self.battery_updated.emit()
        except Exception as exc:
            logger.exception("Battery parse failed: %s", exc)

    def _on_bumper(self, msg: Any) -> None:
        try:
            state = int(msg.state)
            if state == 1:
                idx = int(msg.bumper)
                self.state.fire_bumper(idx)
                self.bump_event.emit(idx, True)
            else:
                self.state.clear_bumpers()
                self.bump_event.emit(-1, False)
        except Exception as exc:
            logger.exception("Bumper parse failed: %s", exc)

    def _on_cliff(self, msg: Any) -> None:
        try:
            state = int(msg.state)
            if state == 1:
                idx = int(msg.sensor)
                self.state.fire_cliff(idx)
                self.cliff_event.emit(idx, True)
            else:
                self.state.clear_cliff()
                self.cliff_event.emit(-1, False)
        except Exception as exc:
            logger.exception("Cliff parse failed: %s", exc)

    def _on_wheel_drop(self, msg: Any) -> None:
        try:
            state = int(msg.state)
            if state == 1:
                idx = int(msg.wheel)
                self.state.fire_wheel_drop(idx)
                self.wheel_drop_event.emit(idx, True)
            else:
                self.state.clear_wheel_drop()
                self.wheel_drop_event.emit(-1, False)
        except Exception as exc:
            logger.exception("WheelDrop parse failed: %s", exc)

    def _on_motion_feedback(self, msg: Any) -> None:
        progress = float(msg.data)
        with self.state.lock:
            self.state.motion_feedback.progress_pct = progress
        self.motion_feedback.emit(progress)

    def _on_motion_result(self, msg: Any) -> None:
        success = bool(msg.data)
        with self.state.lock:
            self.state.motion_feedback.last_result = success
        self.motion_result.emit(success)

    def _on_motion_status(self, msg: Any) -> None:
        status = str(msg.data)
        with self.state.lock:
            self.state.motion_feedback.status = status
        self.motion_status.emit(status)

    def _on_imu_calibrated(self, msg: Any) -> None:
        calibrated = bool(msg.data)
        self.state.set_imu_calibrated(calibrated)
        with self.state.lock:
            progress = self.state.imu_calibration_progress
        self.imu_calibration_changed.emit(calibrated, progress)

    def _on_imu_calibration_progress(self, msg: Any) -> None:
        progress = float(msg.data)
        self.state.set_imu_calibration_progress(progress)
        with self.state.lock:
            calibrated = self.state.imu_calibrated
        self.imu_calibration_changed.emit(calibrated, progress)

    # ---------------------------------------------------------------
    # Publishers
    # ---------------------------------------------------------------

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Publish a continuous teleop velocity to /qbot3/cmd_vel."""
        self._publish_twist("cmd_vel", linear_x, angular_z)
        self.state.set_last_cmd_vel(linear_x, angular_z)

    def publish_precise_cmd(self, distance_m: float = 0.0,
                            angle_rad: float = 0.0) -> None:
        """One-shot closed-loop goal for /qbot3/precise_cmd.

        Set EITHER distance_m (drive forward/back) OR angle_rad (turn CCW/CW).
        Both zero triggers the controller's emergency stop.
        """
        self._publish_twist("precise_cmd", distance_m, angle_rad)

    def publish_emergency_stop(self) -> None:
        """Zero teleop velocity and trip the precise controller's e-stop."""
        self.publish_cmd_vel(0.0, 0.0)
        self._publish_twist("precise_cmd", 0.0, 0.0)

    def publish_reset_odom(self) -> bool:
        """Ask the Pi's odometry_node to zero its pose. Returns True on success."""
        pub = self._publishers.get("odom_reset")
        if pub is None:
            logger.debug("Drop odom reset — bridge not started or topic not configured")
            return False
        try:
            from std_msgs.msg import Bool
            msg = Bool()
            msg.data = True
            pub.publish(msg)
            logger.info("Published /qbot3/odom/reset")
            return True
        except Exception as exc:
            logger.error("Publish odom reset failed: %s", exc)
            return False

    def publish_reset_yaw(self) -> bool:
        """Ask qbot3_base to zero yaw and re-trigger the 5 s gyro calibration.

        The Pi node will:
          * stop motion,
          * clear the gyro-bias sample buffer,
          * sample gyro_z for GYRO_CALIBRATION_DURATION_S,
          * lock in the new bias,
          * resume accepting cmd_vel.
        """
        pub = self._publishers.get("imu_reset_yaw")
        if pub is None:
            logger.debug("Drop yaw reset — bridge not started or topic not configured")
            return False
        try:
            from std_msgs.msg import Bool
            msg = Bool()
            msg.data = True
            pub.publish(msg)
            logger.info("Published /qbot3/imu/reset_yaw — Pi will recalibrate gyro for 5 s")
            return True
        except Exception as exc:
            logger.error("Publish yaw reset failed: %s", exc)
            return False

    def publish_skip_calibration(self) -> bool:
        """Tell qbot3_base to abort the in-progress calibration window.

        Whatever gyro samples it has so far become the bias. If the window
        had just started (no samples yet), bias stays at 0 and online
        refinement during stationary periods will estimate it later.
        """
        pub = self._publishers.get("imu_skip_calibration")
        if pub is None:
            logger.debug("Drop skip calibration — bridge not started or topic not configured")
            return False
        try:
            from std_msgs.msg import Bool
            msg = Bool()
            msg.data = True
            pub.publish(msg)
            logger.info("Published /qbot3/imu/skip_calibration")
            return True
        except Exception as exc:
            logger.error("Publish skip calibration failed: %s", exc)
            return False

    def _publish_twist(self, key: str, linear_x: float, angular_z: float) -> None:
        pub = self._publishers.get(key)
        if pub is None:
            logger.debug("Drop publish to %s — bridge not started", key)
            return
        try:
            from geometry_msgs.msg import Twist
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
            pub.publish(msg)
        except Exception as exc:
            logger.error("Publish %s failed: %s", key, exc)
