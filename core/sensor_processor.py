"""
Sensor Processor — host-side derivations on top of raw sensor data.

The ROS bridge handles transport + raw decode. This module turns those raw
arrays/events into higher-level information used by the AI pipeline,
the SLAM viewer, and the GUI sensor panels:

  * Depth statistics (nearest obstacle, free-space corridor, sector mins)
  * Point-cloud reconstruction from depth + camera intrinsics (Open3D)
  * Latched sensor decay (clear stale bumper/cliff/wheel-drop after N sec)
  * Synthetic data generator for headless / simulation mode

All functions are stateless except for `start_latch_decay_timer` which spawns
a background thread to clear stale latched events.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.shared_state import CameraIntrinsics, SharedState

logger = logging.getLogger(__name__)


# =====================================================================
# Depth analysis
# =====================================================================


@dataclass
class DepthStats:
    """Summary statistics for one depth frame (all distances in meters)."""
    valid_pixel_pct: float
    nearest_distance_m: float
    nearest_direction: str           # "left" | "center" | "right"
    free_corridor_width_px: int
    sector_min_distances_m: List[float]   # left, center-left, center, center-right, right
    histogram_bins_m: List[float]
    histogram_counts: List[int]


def compute_depth_stats(
    depth_mm: np.ndarray,
    *,
    valid_min_mm: int = 100,
    valid_max_mm: int = 5000,
    sectors: int = 5,
) -> Optional[DepthStats]:
    """Summarise a uint16 depth image (millimetres) into navigation-relevant stats."""
    if depth_mm is None or depth_mm.size == 0:
        return None
    if depth_mm.dtype != np.uint16:
        depth_mm = depth_mm.astype(np.uint16)

    valid_mask = (depth_mm >= valid_min_mm) & (depth_mm <= valid_max_mm)
    valid_pct = float(valid_mask.mean()) * 100.0

    # Nearest valid pixel direction
    if valid_mask.any():
        masked = np.where(valid_mask, depth_mm, np.iinfo(np.uint16).max)
        idx = int(np.argmin(masked))
        h, w = depth_mm.shape
        ny, nx = divmod(idx, w)
        nearest_mm = float(depth_mm[ny, nx])
        nearest_distance_m = nearest_mm / 1000.0
        if nx < w / 3:
            nearest_dir = "left"
        elif nx > 2 * w / 3:
            nearest_dir = "right"
        else:
            nearest_dir = "center"
    else:
        nearest_distance_m = float("inf")
        nearest_dir = "center"

    # Sector minimum distances (vertical strips)
    h, w = depth_mm.shape
    sector_mins: List[float] = []
    sector_w = w // sectors
    for i in range(sectors):
        x0 = i * sector_w
        x1 = w if i == sectors - 1 else x0 + sector_w
        strip = depth_mm[:, x0:x1]
        strip_valid = (strip >= valid_min_mm) & (strip <= valid_max_mm)
        if strip_valid.any():
            sector_mins.append(float(strip[strip_valid].min()) / 1000.0)
        else:
            sector_mins.append(float("inf"))

    # Free corridor width: contiguous central columns where the nearest obstacle > 0.6 m
    safe_mm = 600
    col_min = np.where(valid_mask, depth_mm, np.iinfo(np.uint16).max).min(axis=0)
    free_cols = col_min > safe_mm
    if free_cols.any():
        # longest run containing the centre column
        center_col = w // 2
        left = center_col
        while left > 0 and free_cols[left - 1]:
            left -= 1
        right = center_col
        while right < w - 1 and free_cols[right + 1]:
            right += 1
        corridor_width = right - left
    else:
        corridor_width = 0

    # Histogram (0..5 m, 10 bins)
    if valid_mask.any():
        valid_m = depth_mm[valid_mask].astype(np.float32) / 1000.0
        hist, edges = np.histogram(valid_m, bins=10, range=(0.0, 5.0))
        bins = [float((edges[i] + edges[i + 1]) / 2.0) for i in range(len(hist))]
        counts = hist.astype(int).tolist()
    else:
        bins = [0.0] * 10
        counts = [0] * 10

    return DepthStats(
        valid_pixel_pct=valid_pct,
        nearest_distance_m=nearest_distance_m,
        nearest_direction=nearest_dir,
        free_corridor_width_px=corridor_width,
        sector_min_distances_m=sector_mins,
        histogram_bins_m=bins,
        histogram_counts=counts,
    )


def depth_at_pixel(depth_mm: np.ndarray, x: int, y: int, window: int = 5) -> float:
    """Robust depth estimate at (x, y) using the median of a small window. Returns metres."""
    if depth_mm is None:
        return 0.0
    h, w = depth_mm.shape
    half = max(1, window // 2)
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[(patch > 100) & (patch < 5000)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) / 1000.0


def deproject_pixel(intr: CameraIntrinsics, x: int, y: int, depth_m: float
                    ) -> Optional[Tuple[float, float, float]]:
    """Convert a pixel + depth into a 3D point in the camera optical frame."""
    if not intr.is_valid() or depth_m <= 0.0:
        return None
    X = (x - intr.cx) * depth_m / intr.fx
    Y = (y - intr.cy) * depth_m / intr.fy
    Z = depth_m
    return (float(X), float(Y), float(Z))


# =====================================================================
# Depth hole filling — replace zero-depth pixels with the nearest valid
# neighbour. The Pi-side hole filling (pyrealsense2 hole_filling_filter)
# leaves clusters of invalid pixels around foreground edges and on
# texture-less surfaces (white walls, glossy floors). Filling on the host
# means the SLAM grid + 3D cloud + YOLO distance lookups all see a clean
# depth map.
# =====================================================================


import cv2  # noqa: E402  (kept down here so the import is colocated with usage)


def fill_depth_holes(
    depth_mm: np.ndarray,
    *,
    kernel_size: int = 3,
    iterations: int = 4,
    max_fill_distance_px: int = 20,
) -> np.ndarray:
    """Spread valid depth values into zero-depth pixels.

    Algorithm: iteratively replace zeros with the maximum of the surrounding
    valid pixels (cv2.dilate is a max-filter). Each iteration grows the
    valid region by `kernel_size // 2` pixels in each direction; with
    iterations=4 and kernel=3 we fill holes up to ~8 pixels deep, which
    covers the bulk of speckle holes from a RealSense at 640×480.

    Pixels still invalid after `iterations` rounds are left at 0 so callers
    can still distinguish "truly unknown" from "filled estimate".

    Returns a new array; input is not modified.
    """
    if depth_mm is None or depth_mm.size == 0:
        return depth_mm
    if depth_mm.dtype != np.uint16:
        depth_mm = depth_mm.astype(np.uint16)

    invalid = (depth_mm == 0)
    if not invalid.any():
        return depth_mm

    out = depth_mm.copy()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    grown_total = 0
    for _ in range(iterations):
        zeros = (out == 0)
        if not zeros.any():
            break
        # cv2.dilate on uint16 returns the per-window maximum of the
        # neighbourhood. Original valid pixels stay valid; zero pixels
        # adopt the largest neighbour (which is the closer/nearer side
        # for foreground edges, a reasonable estimate).
        dilated = cv2.dilate(out, kernel)
        out = np.where(zeros, dilated, out)
        grown_total += kernel_size // 2
        if grown_total >= max_fill_distance_px:
            break
    return out


# =====================================================================
# Camera mounting transform — shared by SLAM and the 3D point-cloud view
# =====================================================================


def transform_optical_to_base_link(
    points_optical: np.ndarray,
    *,
    camera_height_m: float = 0.0,
    camera_pitch_rad: float = 0.0,
) -> np.ndarray:
    """Convert points from the camera optical frame (Z forward, X right, Y down)
    to the robot base_link frame (X forward, Y left, Z up), applying the
    static camera mounting transform.

    Convention:
        * camera_height_m   — height of the camera optical centre above the floor
        * camera_pitch_rad  — positive = camera tilted nose-down

    Accepts (N, 3) XYZ or (N, 6) XYZRGB; preserves the RGB columns if present.
    """
    if points_optical is None or points_optical.size == 0:
        return points_optical
    X_o = points_optical[:, 0]
    Y_o = points_optical[:, 1]
    Z_o = points_optical[:, 2]

    # Optical → mount frame (camera assumed level for this step):
    #   forward (base X) = Z_optical
    #   left    (base Y) = -X_optical
    #   up      (base Z) = -Y_optical
    X_m = Z_o
    Y_m = -X_o
    Z_m = -Y_o

    # Apply camera pitch (rotation about base_link Y axis, nose-down positive)
    cp = float(np.cos(camera_pitch_rad))
    sp = float(np.sin(camera_pitch_rad))
    X_b = cp * X_m + sp * Z_m
    Y_b = Y_m
    Z_b = -sp * X_m + cp * Z_m + camera_height_m

    out = np.stack([X_b, Y_b, Z_b], axis=1).astype(np.float32)
    if points_optical.shape[1] >= 6:
        out = np.concatenate([out, points_optical[:, 3:6]], axis=1)
    return out


# =====================================================================
# Point cloud reconstruction
# =====================================================================


def reconstruct_point_cloud(
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
    rgb_bgr: Optional[np.ndarray] = None,
    *,
    stride: int = 4,
    max_distance_m: float = 4.0,
) -> Optional[np.ndarray]:
    """Vectorised depth-to-XYZ reconstruction. Returns (N, 6) [x y z r g b] or None.

    Open3D is not used here so this stays cheap and dependency-light; the
    pointcloud_viewer widget can wrap the result in an Open3D PointCloud.
    """
    if depth_mm is None or depth_mm.size == 0 or not intr.is_valid():
        return None

    d = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    h, w = d.shape

    fx = intr.fx / stride
    fy = intr.fy / stride
    cx = intr.cx / stride
    cy = intr.cy / stride

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = (d > 0.05) & (d < max_distance_m)
    if not valid.any():
        return None

    z = d[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    if rgb_bgr is not None and rgb_bgr.shape[:2] == depth_mm.shape:
        rgb = rgb_bgr[::stride, ::stride][valid]
        # BGR -> RGB, normalised 0..1
        rgb = rgb[:, ::-1].astype(np.float32) / 255.0
        return np.concatenate([xyz, rgb], axis=1)
    return xyz


# =====================================================================
# Latched-event decay
# =====================================================================


def start_latch_decay_thread(
    state: SharedState,
    *,
    bumper_timeout_s: float = 1.0,
    cliff_timeout_s: float = 1.0,
    wheel_drop_timeout_s: float = 1.5,
    interval_s: float = 0.25,
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """Background thread that clears stale latched safety events.

    qbot3_base.py publishes events at 10 Hz including state==0 messages,
    so this is mainly defensive — clears latches if the publisher hiccups.
    """
    stop_event = stop_event or threading.Event()

    def _run() -> None:
        while not stop_event.is_set():
            now = time.monotonic()
            with state.lock:
                state.bumpers.expire_older_than(now - bumper_timeout_s)
                state.cliff.expire_older_than(now - cliff_timeout_s)
                state.wheel_drop.expire_older_than(now - wheel_drop_timeout_s)
            stop_event.wait(interval_s)

    t = threading.Thread(target=_run, name="LatchDecay", daemon=True)
    t.start()
    return t


# =====================================================================
# Synthetic data — used when the WebSocket can't reach the Pi
# =====================================================================


class SyntheticSensorGenerator:
    """Drives the SharedState with believable fake data for offline GUI/AI demos."""

    def __init__(self, state: SharedState, *,
                 camera_fps: int = 15, imu_hz: int = 50) -> None:
        self.state = state
        self.camera_period_s = 1.0 / max(1, camera_fps)
        self.imu_period_s = 1.0 / max(1, imu_hz)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._t0 = time.monotonic()

    # -- lifecycle --

    def start(self) -> None:
        if self._threads:
            return
        # Seed an intrinsics matrix so the AI pipeline has something to deproject with
        self.state.set_camera_intrinsics(CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480,
        ))
        self._threads = [
            threading.Thread(target=self._camera_loop, daemon=True, name="SimCamera"),
            threading.Thread(target=self._sensor_loop, daemon=True, name="SimSensors"),
        ]
        for t in self._threads:
            t.start()
        self.state.append_event("INFO", "Simulation mode active — using synthetic sensors")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    # -- generators --

    def _camera_loop(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            rgb = self._make_rgb(t)
            depth = self._make_depth(t)
            self.state.set_rgb_frame(rgb)
            self.state.set_depth_frame(depth)
            self._stop.wait(self.camera_period_s)

    def _sensor_loop(self) -> None:
        x = y = yaw = 0.0
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            t = now - self._t0
            v = 0.05 * math.sin(t * 0.3)
            w = 0.1 * math.sin(t * 0.2)
            yaw += w * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt

            from core.shared_state import IMUReading, OdometryReading, BatteryReading
            self.state.set_imu(IMUReading(
                quat=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                yaw_rad=yaw,
                angular_velocity=(0.0, 0.0, w),
                linear_acceleration=(0.0, 0.0, 9.8),
                monotonic_ts=now,
            ))
            self.state.set_odom(OdometryReading(
                x=x, y=y, yaw_rad=yaw, linear_x=v, angular_z=w, monotonic_ts=now,
            ))
            self.state.set_battery(BatteryReading(
                voltage=12.4 - 0.0001 * t,
                percent=max(0.0, 100.0 - 0.005 * t),
                monotonic_ts=now,
            ))
            self.state.set_encoders(int(x * 2578), int(x * 2578))
            self._stop.wait(self.imu_period_s)

    @staticmethod
    def _make_rgb(t: float) -> np.ndarray:
        """Animated gradient with a moving 'object' so YOLO has something to find."""
        h, w = 480, 640
        # Vertical purple→teal gradient, mild luminance pulse
        gradient = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        purple = np.array([0x6F, 0x63, 0xFF], dtype=np.float32)
        teal = np.array([0xAA, 0xD4, 0x00], dtype=np.float32)
        rgb = (gradient * teal + (1 - gradient) * purple)
        rgb = np.broadcast_to(rgb[:, None, :], (h, w, 3)).astype(np.uint8).copy()
        # Floating "object" disc
        cx = int(w / 2 + 150 * math.sin(t * 0.4))
        cy = int(h / 2 + 60 * math.cos(t * 0.4))
        import cv2
        cv2.circle(rgb, (cx, cy), 50, (200, 200, 250), -1)
        cv2.putText(rgb, "SIM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        return rgb

    @staticmethod
    def _make_depth(t: float) -> np.ndarray:
        """Synthetic depth — far plane with a near 'wall' that wobbles closer/farther."""
        h, w = 480, 640
        depth = np.full((h, w), 3000, dtype=np.uint16)  # 3 m background
        wall_dist = int(800 + 300 * math.sin(t * 0.5))   # 0.5–1.1 m wobble
        depth[h - 80:h - 20, :] = wall_dist
        return depth
