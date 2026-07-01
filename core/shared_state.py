"""
Shared State — single source of truth for the host-side application.

This module is intentionally Qt-agnostic. All inputs land here from the ROS
bridge, the AI pipeline, the SLAM manager, and the active mode. All consumers
(GUI widgets, skills, planners) read from here.

Thread safety:
    A single `threading.RLock` protects every mutable field. Readers and
    writers must enter the lock — use the `state.lock` context manager for
    compound updates, or call the dedicated setters/getters which lock once.

Signals are NOT emitted from this class. The ROS bridge owns the Qt signal
layer; this module only stores data.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Dataclasses for structured sensor / AI / mode data
# =====================================================================


@dataclass
class IMUReading:
    """IMU sample. Orientation is a normalized quaternion (x, y, z, w)."""
    quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    yaw_rad: float = 0.0
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: Tuple[float, float, float] = (0.0, 0.0, 9.8)
    monotonic_ts: float = 0.0


@dataclass
class OdometryReading:
    """Pose + twist from /qbot3/odom (frame `odom` -> `base_link`)."""
    x: float = 0.0
    y: float = 0.0
    yaw_rad: float = 0.0
    linear_x: float = 0.0
    angular_z: float = 0.0
    monotonic_ts: float = 0.0


@dataclass
class BatteryReading:
    """Battery voltage + computed percentage."""
    voltage: float = 0.0
    percent: float = 0.0
    monotonic_ts: float = 0.0


@dataclass
class LatchedTriState:
    """Latched state for sensors with left/center/right (or left/right) channels.

    Channels are stored in a fixed-length list of floats holding the
    monotonic timestamp at which each channel last fired (0.0 = never / clear).
    """
    timestamps: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def fire(self, idx: int, ts: float) -> None:
        if 0 <= idx < len(self.timestamps):
            self.timestamps[idx] = ts

    def clear_all(self) -> None:
        for i in range(len(self.timestamps)):
            self.timestamps[i] = 0.0

    def expire_older_than(self, cutoff_ts: float) -> None:
        for i, ts in enumerate(self.timestamps):
            if ts and ts < cutoff_ts:
                self.timestamps[i] = 0.0

    def active(self) -> List[bool]:
        return [ts > 0.0 for ts in self.timestamps]

    def any_active(self) -> bool:
        return any(ts > 0.0 for ts in self.timestamps)


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics from /camera/color/camera_info."""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 640
    height: int = 480

    def is_valid(self) -> bool:
        return self.fx > 0.0 and self.fy > 0.0


@dataclass
class Detection:
    """A single YOLO detection enriched with depth/3D info."""
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy: Tuple[int, int]
    distance_m: float = 0.0
    position_3d: Optional[Tuple[float, float, float]] = None
    monotonic_ts: float = 0.0


@dataclass
class VLMOutput:
    """Structured VLM (scene-understanding) result."""
    scene_description: str = ""
    object_relationships: str = ""
    navigation_hints: str = ""
    task_observations: str = ""
    raw_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    monotonic_ts: float = 0.0


@dataclass
class LLMOutput:
    """Structured LLM (planner) result — mirrors the planner JSON schema."""
    reasoning: str = ""
    confidence: float = 0.0
    action_type: str = ""               # "low_level" | "skill"
    low_level_command: Optional[Dict[str, Any]] = None
    skill_command: Optional[Dict[str, Any]] = None
    status: str = ""                    # "executing" | "task_complete" | ...
    next_observation: str = ""
    raw_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    monotonic_ts: float = 0.0


@dataclass
class TaskRecord:
    """One entry in the user-facing task history."""
    description: str
    started_ts: float
    finished_ts: Optional[float] = None
    status: str = "running"             # "running" | "success" | "failed" | "cancelled"
    notes: str = ""


@dataclass
class SkillRecord:
    """One entry in the skill execution history."""
    name: str
    started_ts: float
    finished_ts: Optional[float] = None
    success: Optional[bool] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionFeedback:
    """Latest closed-loop motion controller feedback."""
    progress_pct: float = 0.0
    status: str = "idle"                # "idle" | "moving" | "turning" | "emergency_stop"
    last_result: Optional[bool] = None


# =====================================================================
# SharedState singleton
# =====================================================================


class SharedState:
    """Thread-safe singleton storing all live application state."""

    _instance: Optional["SharedState"] = None
    _instance_lock = threading.Lock()

    # -- singleton accessor --
    @classmethod
    def instance(cls) -> "SharedState":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = SharedState()
        return cls._instance

    def __init__(self) -> None:
        # Re-entrant so the state can be locked across nested helper calls
        self.lock = threading.RLock()

        # ---- Connection ----
        self.ros_connected: bool = False
        self.ros_status_message: str = "disconnected"

        # ---- Vision ----
        self.rgb_frame: Optional[np.ndarray] = None        # BGR uint8 H×W×3
        self.depth_frame: Optional[np.ndarray] = None      # uint16 mm
        self.depth_visual_frame: Optional[np.ndarray] = None  # BGR uint8 (Pi-side colorized)
        self.point_cloud: Optional[np.ndarray] = None      # (N,3) float32, meters
        self.camera_intrinsics: CameraIntrinsics = CameraIntrinsics()
        self.last_rgb_ts: float = 0.0
        self.last_depth_ts: float = 0.0

        # ---- Detections ----
        self.detected_objects: List[Detection] = []

        # ---- Robot state ----
        self.imu: IMUReading = IMUReading()
        self.odom: OdometryReading = OdometryReading()
        self.battery: BatteryReading = BatteryReading()
        # Pi-side gyro-bias calibration status (5 s startup window).
        # imu_calibrated transitions False → True once the Pi has finished
        # estimating bias; UI banner + mission-start gate watch this.
        self.imu_calibrated: bool = False
        self.imu_calibration_progress: float = 0.0
        self.bumpers: LatchedTriState = LatchedTriState([0.0, 0.0, 0.0])      # L, C, R
        self.cliff: LatchedTriState = LatchedTriState([0.0, 0.0, 0.0])        # L, C, R
        self.wheel_drop: LatchedTriState = LatchedTriState([0.0, 0.0])        # L, R
        self.encoders_lr: Tuple[int, int] = (0, 0)
        self.last_cmd_vel: Tuple[float, float] = (0.0, 0.0)
        self.motion_feedback: MotionFeedback = MotionFeedback()

        # ---- AI outputs ----
        self.vlm_last_output: Optional[VLMOutput] = None
        self.llm_last_output: Optional[LLMOutput] = None

        # ---- Mode + skills ----
        self.active_mode: str = "manual"                  # "ai" | "manual" | "skills"
        self.active_skill: Optional[str] = None
        self.active_skill_progress: float = 0.0
        self.active_task_description: str = ""

        # ---- History (capped) ----
        self.task_history: Deque[TaskRecord] = deque(maxlen=64)
        self.skill_history: Deque[SkillRecord] = deque(maxlen=64)
        self.action_history: Deque[Dict[str, Any]] = deque(maxlen=32)
        self.event_log: Deque[Tuple[float, str, str]] = deque(maxlen=512)  # (ts, level, msg)

        # ---- Map / waypoints ----
        self.slam_map: Optional[np.ndarray] = None        # 2D occupancy grid
        self.slam_origin: Tuple[float, float, float] = (0.0, 0.0, 0.05)  # (x, y, resolution m/cell)
        self.slam_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # x, y, theta
        self.slam_trajectory: Deque[Tuple[float, float]] = deque(maxlen=4096)
        self.manual_waypoints: List[Tuple[float, float, float]] = []
        self.named_waypoints: Dict[str, Tuple[float, float, float]] = {}

        # ---- Object memory (RAG) ----
        # Attached by main.py once the persistent store path is known. Skills
        # and the planner read from here to look up remembered objects by
        # class name. None until wired so unit tests / GUI smoke don't need it.
        self.object_memory: Optional[Any] = None

        logger.debug("SharedState singleton initialized")

    # =================================================================
    # Convenience setters (each takes the lock for an atomic update)
    # =================================================================

    def set_rgb_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.rgb_frame = frame
            self.last_rgb_ts = time.monotonic()

    def set_depth_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.depth_frame = frame
            self.last_depth_ts = time.monotonic()

    def set_depth_visual(self, frame: np.ndarray) -> None:
        with self.lock:
            self.depth_visual_frame = frame

    def set_point_cloud(self, points: np.ndarray) -> None:
        with self.lock:
            self.point_cloud = points

    def set_camera_intrinsics(self, intr: CameraIntrinsics) -> None:
        with self.lock:
            self.camera_intrinsics = intr

    def set_imu(self, reading: IMUReading) -> None:
        with self.lock:
            self.imu = reading

    def set_imu_calibrated(self, calibrated: bool) -> None:
        with self.lock:
            self.imu_calibrated = bool(calibrated)

    def set_imu_calibration_progress(self, progress: float) -> None:
        with self.lock:
            self.imu_calibration_progress = max(0.0, min(1.0, float(progress)))

    def set_odom(self, reading: OdometryReading) -> None:
        with self.lock:
            self.odom = reading
            self.slam_pose = (reading.x, reading.y, reading.yaw_rad)
            self.slam_trajectory.append((reading.x, reading.y))

    def set_battery(self, reading: BatteryReading) -> None:
        with self.lock:
            self.battery = reading

    def set_encoders(self, left: int, right: int) -> None:
        with self.lock:
            self.encoders_lr = (left, right)

    def set_detections(self, detections: List[Detection]) -> None:
        with self.lock:
            self.detected_objects = detections

    def set_vlm_output(self, output: VLMOutput) -> None:
        with self.lock:
            self.vlm_last_output = output

    def set_llm_output(self, output: LLMOutput) -> None:
        with self.lock:
            self.llm_last_output = output

    def set_active_mode(self, mode: str) -> None:
        if mode not in ("ai", "manual", "skills"):
            logger.warning("Ignoring unknown mode: %s", mode)
            return
        with self.lock:
            self.active_mode = mode
            self.event_log.append((time.monotonic(), "INFO", f"Mode -> {mode}"))

    def set_connection(self, connected: bool, message: str = "") -> None:
        with self.lock:
            self.ros_connected = connected
            self.ros_status_message = message or ("connected" if connected else "disconnected")

    def set_last_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        with self.lock:
            self.last_cmd_vel = (linear_x, angular_z)

    def set_motion_feedback(self, feedback: MotionFeedback) -> None:
        with self.lock:
            self.motion_feedback = feedback

    # ---- Latched event helpers ----

    def fire_bumper(self, sensor_idx: int) -> None:
        with self.lock:
            self.bumpers.fire(sensor_idx, time.monotonic())

    def clear_bumpers(self) -> None:
        with self.lock:
            self.bumpers.clear_all()

    def fire_cliff(self, sensor_idx: int) -> None:
        with self.lock:
            self.cliff.fire(sensor_idx, time.monotonic())

    def clear_cliff(self) -> None:
        with self.lock:
            self.cliff.clear_all()

    def fire_wheel_drop(self, wheel_idx: int) -> None:
        with self.lock:
            self.wheel_drop.fire(wheel_idx, time.monotonic())

    def clear_wheel_drop(self) -> None:
        with self.lock:
            self.wheel_drop.clear_all()

    # ---- History / waypoints ----

    def append_action(self, action: Dict[str, Any]) -> None:
        action = dict(action)
        action.setdefault("ts", time.monotonic())
        with self.lock:
            self.action_history.append(action)

    def append_task(self, task: TaskRecord) -> None:
        with self.lock:
            self.task_history.append(task)

    def append_skill(self, skill: SkillRecord) -> None:
        with self.lock:
            self.skill_history.append(skill)

    def append_event(self, level: str, message: str) -> None:
        with self.lock:
            self.event_log.append((time.monotonic(), level, message))

    def append_manual_waypoint(self) -> None:
        """Snapshot current odom into the manual-driving waypoint list."""
        with self.lock:
            wp = (self.odom.x, self.odom.y, self.odom.yaw_rad)
            if not self.manual_waypoints or _far_enough(self.manual_waypoints[-1], wp):
                self.manual_waypoints.append(wp)

    def save_named_waypoint(self, name: str) -> None:
        with self.lock:
            self.named_waypoints[name] = (
                self.odom.x, self.odom.y, self.odom.yaw_rad,
            )
            self.event_log.append(
                (time.monotonic(), "INFO", f"Saved waypoint '{name}'")
            )

    # ---- Snapshot helpers (for AI context assembly) ----

    def snapshot_for_planner(self) -> Dict[str, Any]:
        """Lock-free read isn't safe — produce a deep enough snapshot dict."""
        with self.lock:
            return {
                "odom": {
                    "x": self.odom.x,
                    "y": self.odom.y,
                    "yaw_rad": self.odom.yaw_rad,
                },
                "battery_percent": self.battery.percent,
                "bumpers_active": self.bumpers.any_active(),
                "cliff_active": self.cliff.any_active(),
                "wheel_drop_active": self.wheel_drop.any_active(),
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "distance_m": d.distance_m,
                        "position_3d": d.position_3d,
                    }
                    for d in self.detected_objects
                ],
                "named_waypoints": dict(self.named_waypoints),
                "action_history": list(self.action_history),
                "manual_waypoints_count": len(self.manual_waypoints),
                "active_mode": self.active_mode,
                "active_skill": self.active_skill,
                "remembered_objects": (
                    self.object_memory.format_for_planner()
                    if self.object_memory is not None else "(none)"
                ),
            }


def _far_enough(a: Tuple[float, float, float], b: Tuple[float, float, float],
                min_dist_m: float = 0.05) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) >= (min_dist_m * min_dist_m)
