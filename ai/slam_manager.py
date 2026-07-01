"""
SLAM Manager — host-side 2D occupancy grid with optional RTAB-Map ingest.

Two operating modes:
    HOST_SIMPLE   (default) — build a log-odds occupancy grid on the host PC
                              from /qbot3/odom + /camera/depth/raw + intrinsics.
                              Works without anything extra on the Pi.
    EXTERNAL      (opt-in)  — subscribe to /map (nav_msgs/OccupancyGrid) if
                              RTAB-Map is launched on the Pi. Replaces the
                              host grid when messages arrive.

Public API (used by the GUI SLAM viewer + skills + LLM context):
    start() / stop()
    get_map_image()                  -> np.ndarray  H×W×3 BGR (rendered for display)
    get_pose()                       -> (x, y, theta_rad)
    get_nearest_obstacle_distance()  -> float meters
    get_free_space_map()             -> np.ndarray  H×W bool
    world_to_pixel(x, y)             -> (col, row)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.shared_state import CameraIntrinsics, SharedState

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================


class SlamMode(str, Enum):
    HOST_SIMPLE = "host_simple"
    EXTERNAL = "external"
    IDLE = "idle"


# Log-odds clip range — Round 21 retune so depth noise can't dominate.
# A single hit bumps a cell by 0.70 (was 1.20). A cell needs ~3 confirming
# frames to cross PROB_OCC and render as a wall, which is what stops a
# stray floor pixel or motion-blur artefact from being mistaken for a
# wall. FREE bumps are −0.70 (was −0.50) so the room interior gets
# cleared faster than spurious obstacles can accumulate.
LOG_ODDS_OCCUPIED = 0.70
LOG_ODDS_FREE = -0.70
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0

# Display thresholds — Round 21 stricter so only confidently-occupied
# cells render as wall. PROB_OCC 0.55 → 0.65 means a cell must hit
# log_odds ≈ +0.62 (a few confirming frames) before turning black.
PROB_OCC = 0.65
PROB_FREE = 0.40

# ---- Render cleanup (Round 21) ----
# Round 20's OPEN+DILATE morphology was erasing legit 1-cell wall
# segments — exactly the thin laser-scan-style traces the user wants.
# We now CLOSE the FREE mask only (so the room interior reads as a
# clean white blob) and leave the OCC mask untouched (so walls render
# at their true 1-cell thickness). The MORPH_OPEN pass is disabled
# entirely; speckle noise gets filtered out at the SOURCE by the
# stricter log-odds threshold above.
RENDER_CLOSE_KERNEL = 2        # cells — small CLOSE on FREE only
RENDER_OPEN_KERNEL = 0         # disabled (was 3 in Round 20)
RENDER_DILATE_OCC = 0          # disabled — walls stay 1-px thin
# Final blit upscale factor — the underlying grid is rendered at
# (size_cells × size_cells), then enlarged via LANCZOS4 so edges stay
# crisp on a typical SLAM-tab viewport.
RENDER_UPSCALE = 2

# Rendering colours (BGR, **standard 2-D occupancy-grid convention** —
# light theme to match RViz / gmapping / RTAB-Map output the operator
# referenced in Round 21):
#   white      → free space the robot has cleared
#   light-grey → unknown / unmapped
#   near-black → confirmed walls / obstacles
COLOR_UNKNOWN = (210, 210, 218)      # light grey background
COLOR_FREE = (250, 250, 252)         # near-white free space
COLOR_OCC = (20, 22, 30)             # near-black walls
COLOR_GRID = (165, 165, 175)         # subtle 1 m grid lines
COLOR_TRAIL = (40, 165, 90)          # darker green so it pops on white
COLOR_ROBOT = (120, 90, 220)         # purple ring
COLOR_ROBOT_BODY = (140, 110, 230)   # purple filled body
COLOR_FOV = (200, 175, 255)          # soft purple FOV cone
COLOR_WAYPOINT = (50, 50, 200)
COLOR_NAMED_WAYPOINT = (10, 130, 230)
COLOR_OBJECT = (40, 80, 220)         # red-orange object marker
COLOR_HUD_TEXT = (40, 40, 55)
COLOR_HUD_DIM = (110, 110, 125)


# =====================================================================
# SlamManager
# =====================================================================


class SlamManager(QObject):
    """Builds and serves a 2D occupancy grid for the rest of the app."""

    map_updated = pyqtSignal()

    def __init__(
        self,
        state: Optional[SharedState] = None,
        ros_bridge: Optional[Any] = None,
        *,
        size_cells: int = 400,
        resolution_m: float = 0.05,
        update_hz: float = 8.0,        # Round 21: 5 → 8 Hz so accumulation feels live
        max_range_m: float = 4.0,
        # Camera mounting calibration (read from config/settings.json -> calibration.*)
        camera_height_m: float = 0.10,
        camera_pitch_deg: float = 0.0,
        # Round 21: narrower obstacle window. The 0.02–2.00 m range was
        # capturing floor speckle (low pixels at near range) and ceiling
        # features. 0.15–1.80 m drops both, leaving only solid wall /
        # furniture features in the obstacle pass.
        obstacle_min_height_m: float = 0.15,
        obstacle_max_height_m: float = 1.80,
        # When |angular velocity| exceeds this, skip the integration step.
        # Loosened so we don't skip too often — real obstacles are missed
        # when this threshold is too tight.
        max_angular_velocity_radps: float = 1.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state or SharedState.instance()
        self._ros_bridge = ros_bridge

        self.mode: SlamMode = SlamMode.HOST_SIMPLE
        self.size = size_cells
        self.resolution = resolution_m
        self.max_range_m = max_range_m

        # Camera mounting (radians for the math)
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_rad = math.radians(float(camera_pitch_deg))
        self.obstacle_min_height_m = float(obstacle_min_height_m)
        self.obstacle_max_height_m = float(obstacle_max_height_m)
        self.max_angular_velocity_radps = float(max_angular_velocity_radps)

        # Log-odds grid centred on (0, 0); origin = grid centre
        self._lock = threading.RLock()
        self._log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self._origin_cell = (self.size // 2, self.size // 2)
        self._last_render: Optional[np.ndarray] = None
        self._external_grid: Optional[np.ndarray] = None
        self._external_origin: Tuple[float, float, float] = (0.0, 0.0, self.resolution)
        self._external_subscriber: Optional[Any] = None

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / max(0.5, update_hz)))
        self._timer.timeout.connect(self._tick)

        # Pre-compute a pixel mesh that we'll use for ray-casting from depth
        self._du: Optional[np.ndarray] = None
        self._dv: Optional[np.ndarray] = None
        self._mesh_shape: Tuple[int, int] = (0, 0)
        self._frames_integrated: int = 0
        self._frames_skipped_motion: int = 0

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def start(self, mode: SlamMode = SlamMode.HOST_SIMPLE) -> None:
        self.mode = mode
        if mode == SlamMode.EXTERNAL:
            self._subscribe_external_map()
        self._timer.start()
        logger.info("SlamManager started (mode=%s, %dx%d cells @ %.2f m)",
                    mode.value, self.size, self.size, self.resolution)

    def stop(self) -> None:
        self._timer.stop()
        if self._external_subscriber is not None:
            try:
                self._external_subscriber.unsubscribe()
            except Exception:
                pass
            self._external_subscriber = None

    def reset(self) -> None:
        with self._lock:
            self._log_odds.fill(0.0)
            self._external_grid = None
            self._last_render = None
        self.state.slam_trajectory.clear()

    # ---------------------------------------------------------------
    # Public read-only API
    # ---------------------------------------------------------------

    def get_pose(self) -> Tuple[float, float, float]:
        with self.state.lock:
            o = self.state.odom
            return (o.x, o.y, o.yaw_rad)

    def get_free_space_map(self) -> np.ndarray:
        """Boolean map: True where probability of free > PROB_FREE."""
        with self._lock:
            if self._external_grid is not None:
                return self._external_grid == 0
            prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
            return prob < PROB_FREE

    def get_nearest_obstacle_distance(self) -> float:
        with self._lock:
            if self._external_grid is not None:
                occ = self._external_grid > 50
            else:
                prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
                occ = prob > PROB_OCC
            if not occ.any():
                return float("inf")
            rx, ry, _ = self.get_pose()
            cr, cc = self._world_to_cell(rx, ry)
            ys, xs = np.where(occ)
            d_cells = np.sqrt((ys - cr) ** 2 + (xs - cc) ** 2)
            return float(d_cells.min() * self.resolution)

    def get_map_image(self) -> np.ndarray:
        with self._lock:
            if self._last_render is None:
                self._render_locked()
            return self._last_render.copy() if self._last_render is not None else np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        cr, cc = self._world_to_cell(x, y)
        return (int(cc), int(self.size - 1 - cr))   # cv2 (col, row), image is flipped vertically

    # ---------------------------------------------------------------
    # Internal — periodic update + integration
    # ---------------------------------------------------------------

    def _tick(self) -> None:
        try:
            if self.mode == SlamMode.HOST_SIMPLE:
                self._integrate_depth_frame()
            with self._lock:
                self._render_locked()
        except Exception as exc:
            logger.exception("SLAM tick failed: %s", exc)
        else:
            self.map_updated.emit()

    def _integrate_depth_frame(self) -> None:
        """Project depth pixels into the world frame using full camera mounting
        transform, then keep only points whose world-Z falls inside the
        obstacle height window. Skips integration when the robot is turning
        fast enough that motion blur + odom-yaw lag would smear obstacles
        into bogus arcs.
        """
        with self.state.lock:
            depth = self.state.depth_frame
            intr = self.state.camera_intrinsics
            pose_x = self.state.odom.x
            pose_y = self.state.odom.y
            pose_yaw = self.state.odom.yaw_rad
            ang_z = abs(self.state.imu.angular_velocity[2])
        if depth is None or not intr.is_valid():
            return

        # Reject frames captured during a fast turn — they map to bogus arcs.
        if ang_z > self.max_angular_velocity_radps:
            self._frames_skipped_motion += 1
            return

        # Subsample for speed (8x in each dim → 80×60 = 4800 candidate pixels)
        stride = 8
        d = depth[::stride, ::stride].astype(np.float32) / 1000.0
        h, w = d.shape
        if (h, w) != self._mesh_shape:
            us = np.arange(w) * stride
            vs = np.arange(h) * stride
            self._du, self._dv = np.meshgrid(us.astype(np.float32), vs.astype(np.float32))
            self._mesh_shape = (h, w)

        valid = (d > 0.05) & (d < self.max_range_m)
        if not valid.any():
            return

        # ---- Camera optical frame coords (Z forward, X right, Y down) ----
        z_o = d[valid]
        x_o = (self._du[valid] - intr.cx) * z_o / intr.fx
        y_o = (self._dv[valid] - intr.cy) * z_o / intr.fy

        # ---- Optical → mount frame (X forward, Y left, Z up; level camera) ----
        X_m = z_o
        Y_m = -x_o
        Z_m = -y_o

        # ---- Apply camera pitch (rotation about Y, nose-down positive) ----
        cp = math.cos(self.camera_pitch_rad)
        sp = math.sin(self.camera_pitch_rad)
        X_b = cp * X_m + sp * Z_m
        Y_b = Y_m
        Z_b = -sp * X_m + cp * Z_m + self.camera_height_m

        # ---- Filter to obstacle height window (in metres above floor) ----
        obstacle = (Z_b >= self.obstacle_min_height_m) & (Z_b <= self.obstacle_max_height_m)
        if not obstacle.any():
            return
        X_b = X_b[obstacle]
        Y_b = Y_b[obstacle]

        # ---- Apply robot yaw → world XY ----
        cy = math.cos(pose_yaw)
        sy = math.sin(pose_yaw)
        wx = pose_x + cy * X_b - sy * Y_b
        wy = pose_y + sy * X_b + cy * Y_b

        with self._lock:
            # Mark obstacle cells as occupied (log-odds bump)
            cells = self._world_array_to_cells(wx, wy)
            self._bump_cells(cells, LOG_ODDS_OCCUPIED)

            # Mark a sparse free-space line between robot and each hit
            self._mark_free_rays(pose_x, pose_y, wx, wy, max_steps=20)
        self._frames_integrated += 1

    def _mark_free_rays(self, x0: float, y0: float,
                        xs: np.ndarray, ys: np.ndarray, *, max_steps: int) -> None:
        # Denser free-ray pass (Round 20) — was 200 rays / 20 steps, now
        # 400 rays / 28 steps. Free space fills in roughly 2.8× faster
        # per cycle, which is what cured the "scan is too slow" complaint
        # without inflating CPU enough to matter (vector ops, no Python loop
        # over individual cells).
        stride = max(1, xs.size // 400)
        xs = xs[::stride]
        ys = ys[::stride]
        steps = max(max_steps, 28)
        if xs.size == 0:
            return
        # Vectorised: build the full (rays, steps) grid in one shot.
        ts = np.linspace(0.0, 1.0, steps + 1)[:-1]      # exclude endpoint
        xs_line = (x0 + (xs[:, None] - x0) * ts[None, :]).reshape(-1)
        ys_line = (y0 + (ys[:, None] - y0) * ts[None, :]).reshape(-1)
        cells = self._world_array_to_cells(xs_line, ys_line)
        self._bump_cells(cells, LOG_ODDS_FREE)

    # ---------------------------------------------------------------
    # Public mutation API (used by bumper handler in BaseSkill and the
    # explore_room skill)
    # ---------------------------------------------------------------

    def stamp_obstacle(self, x: float, y: float, *,
                       radius_m: float = 0.20,
                       weight: float = 4.0) -> None:
        """Mark a disc on the occupancy grid centred at world (x, y).
        Used when a bumper fires — we KNOW there's a wall at the robot's
        front face, but depth-based integration may have missed it (the
        camera looks slightly upward / it's too close to focus).

        `radius_m` is the half-width of the marked patch in metres;
        `weight` is the log-odds bump applied at every cell (clamped by
        LOG_ODDS_MAX). A single call with radius_m=0.20, weight=4.0
        produces a confidently-occupied disc the next render cycle.
        """
        if radius_m <= 0:
            return
        radius_cells = max(1, int(round(radius_m / self.resolution)))
        with self._lock:
            cr0, cc0 = self._origin_cell
            cc = int(round(cc0 + x / self.resolution))
            cr = int(round(cr0 + y / self.resolution))
            r0 = max(0, cr - radius_cells)
            r1 = min(self.size, cr + radius_cells + 1)
            c0 = max(0, cc - radius_cells)
            c1 = min(self.size, cc + radius_cells + 1)
            if r1 <= r0 or c1 <= c0:
                return
            ys, xs = np.ogrid[r0:r1, c0:c1]
            mask = (ys - cr) ** 2 + (xs - cc) ** 2 <= radius_cells ** 2
            self._log_odds[r0:r1, c0:c1][mask] += weight
            np.clip(self._log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX,
                    out=self._log_odds)
            self._last_render = None     # force re-render next tick
        logger.info("SLAM: stamped obstacle at world (%.2f, %.2f) r=%.2fm",
                    x, y, radius_m)

    def get_frontier_cells(self) -> np.ndarray:
        """Return cell (row, col) indices of "frontiers" — known-free
        cells that touch at least one unknown cell. The new
        `explore_room` skill drives toward the closest frontier whose
        target-belief weighting is highest.

        Returns an (N, 2) int32 array in (row, col) form. Empty if the
        grid is all-unknown or all-known.
        """
        with self._lock:
            if self._external_grid is not None:
                # OccupancyGrid: 0 free, -1 unknown, >0 occupied
                grid = self._external_grid
                free = grid == 0
                unknown = grid < 0
            else:
                prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
                free = prob < PROB_FREE
                unknown = (prob >= PROB_FREE) & (prob <= PROB_OCC)
        if not free.any() or not unknown.any():
            return np.zeros((0, 2), dtype=np.int32)
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(unknown.astype(np.uint8), kernel) > 0
        frontier = free & unknown_dilated
        ys, xs = np.where(frontier)
        return np.stack([ys.astype(np.int32), xs.astype(np.int32)], axis=1)

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Inverse of _world_to_cell — used by explore_room to convert
        a chosen frontier cell back into a (x, y) drive target."""
        cr0, cc0 = self._origin_cell
        x = (col - cc0) * self.resolution
        y = (row - cr0) * self.resolution
        return (x, y)

    def is_free(self, x: float, y: float) -> bool:
        """True if world (x, y) sits in a known-free cell. Used by the
        safe-viewpoint planner in vlm_reach."""
        with self._lock:
            cr, cc = self._world_to_cell(x, y)
            if self._external_grid is not None:
                return bool(self._external_grid[cr, cc] == 0)
            prob = 1.0 - 1.0 / (1.0 + math.exp(self._log_odds[cr, cc]))
            return prob < PROB_FREE

    def _bump_cells(self, cells: Tuple[np.ndarray, np.ndarray], delta: float) -> None:
        rows, cols = cells
        if rows.size == 0:
            return
        np.add.at(self._log_odds, (rows, cols), delta)
        np.clip(self._log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self._log_odds)

    def _world_array_to_cells(self, xs: np.ndarray, ys: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        cr0, cc0 = self._origin_cell
        cols = np.clip(np.round(cc0 + xs / self.resolution), 0, self.size - 1).astype(np.int32)
        rows = np.clip(np.round(cr0 + ys / self.resolution), 0, self.size - 1).astype(np.int32)
        return rows, cols

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cr0, cc0 = self._origin_cell
        cc = int(round(cc0 + x / self.resolution))
        cr = int(round(cr0 + y / self.resolution))
        cc = max(0, min(self.size - 1, cc))
        cr = max(0, min(self.size - 1, cr))
        return cr, cc

    # ---------------------------------------------------------------
    # External RTAB-Map ingestion
    # ---------------------------------------------------------------

    def _subscribe_external_map(self) -> None:
        bridge = self._ros_bridge
        if bridge is None:
            logger.warning("EXTERNAL slam mode requested but no ros_bridge — falling back to HOST_SIMPLE")
            self.mode = SlamMode.HOST_SIMPLE
            return
        client = getattr(bridge, "_client", None)
        if client is None:
            logger.warning("ros_bridge has no client yet — external map sub deferred")
            return
        try:
            import roslibpy
            sub = roslibpy.Topic(
                client, "/map", "nav_msgs/OccupancyGrid",
                queue_length=1, compression="cbor-raw",
            )
            sub.subscribe(self._on_external_map)
            self._external_subscriber = sub
            logger.info("Subscribed to /map for external SLAM ingestion")
        except Exception as exc:
            logger.warning("External map sub failed: %s — falling back", exc)
            self.mode = SlamMode.HOST_SIMPLE

    def _on_external_map(self, msg: Dict[str, Any]) -> None:
        try:
            info = msg.get("info", {})
            width = int(info.get("width", 0))
            height = int(info.get("height", 0))
            resolution = float(info.get("resolution", self.resolution))
            origin = info.get("origin", {}).get("position", {})
            data = msg.get("data") or []
            if width == 0 or height == 0 or len(data) != width * height:
                return
            grid = np.array(data, dtype=np.int8).reshape(height, width)
            with self._lock:
                self._external_grid = grid
                self._external_origin = (
                    float(origin.get("x", 0.0)),
                    float(origin.get("y", 0.0)),
                    resolution,
                )
        except Exception as exc:
            logger.exception("External /map parse failed: %s", exc)

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    @staticmethod
    def _clean_masks(occ_mask: np.ndarray, free_mask: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Light morphology pass — Round 21 tuned to preserve thin
        single-cell walls (the laser-scan look the operator wants).

        * FREE mask CLOSE  — bridge any 1-cell gaps in cleared corridors
          so the interior reads as a single solid white blob, not a
          dotty pattern.
        * OCC  mask LEFT ALONE — speckle is already filtered out at the
          source by the stricter LOG_ODDS_OCCUPIED + PROB_OCC pair, and
          aggressive OPEN/DILATE was erasing legit thin walls in
          Round 20 (single-cell laser returns from a long thin wall got
          deleted as "speckle").
        * Occupied wins where the two masks overlap.
        """
        if RENDER_CLOSE_KERNEL > 1:
            k = max(1, RENDER_CLOSE_KERNEL)
            kc = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            free_clean = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, kc)
        else:
            free_clean = free_mask
        occ_clean = occ_mask
        # Optional speckle filter — disabled by default (kernel=0) so we
        # keep thin walls. Re-enable by bumping RENDER_OPEN_KERNEL only
        # if the source data is unusually noisy.
        if RENDER_OPEN_KERNEL > 1:
            ko = cv2.getStructuringElement(
                cv2.MORPH_RECT, (RENDER_OPEN_KERNEL, RENDER_OPEN_KERNEL),
            )
            occ_clean = cv2.morphologyEx(occ_clean, cv2.MORPH_OPEN, ko)
        if RENDER_DILATE_OCC > 0:
            kd = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1 + 2 * RENDER_DILATE_OCC, 1 + 2 * RENDER_DILATE_OCC),
            )
            occ_clean = cv2.dilate(occ_clean, kd)
        # Occupied wins when both fire (a wall cell shouldn't read as free).
        free_clean = free_clean & (1 - occ_clean)
        return occ_clean.astype(np.uint8), free_clean.astype(np.uint8)

    def _render_locked(self) -> None:
        # ---- 1. Base layer: occupancy from log-odds (or external grid) ----
        # Build the occupied + free binary masks first, clean them with a
        # small morphological pipeline (CLOSE → OPEN → mild DILATE on
        # occupied), THEN colour the result. This is what turns the raw
        # speckled grey grid into a crisp map.
        if self._external_grid is not None:
            grid = self._external_grid
            occ_mask = (grid > 50).astype(np.uint8)
            free_mask = (grid == 0).astype(np.uint8)
        else:
            prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
            occ_mask = (prob > PROB_OCC).astype(np.uint8)
            free_mask = (prob < PROB_FREE).astype(np.uint8)

        occ_mask, free_mask = self._clean_masks(occ_mask, free_mask)
        occ_count = int(occ_mask.sum())

        img = np.full((occ_mask.shape[0], occ_mask.shape[1], 3),
                      COLOR_UNKNOWN, dtype=np.uint8)
        img[free_mask.astype(bool)] = COLOR_FREE
        img[occ_mask.astype(bool)] = COLOR_OCC
        img = cv2.flip(img, 0)

        h, w = img.shape[:2]

        # ---- 2. 1 m grid lines for spatial reference ----
        cells_per_meter = int(round(1.0 / self.resolution))
        if cells_per_meter > 4:
            for i in range(0, w, cells_per_meter):
                cv2.line(img, (i, 0), (i, h - 1), COLOR_GRID, 1, lineType=cv2.LINE_8)
            for i in range(0, h, cells_per_meter):
                cv2.line(img, (0, i), (w - 1, i), COLOR_GRID, 1, lineType=cv2.LINE_8)

        # ---- 3. Trajectory polyline (thicker so it's actually visible) ----
        traj = list(self.state.slam_trajectory)
        if len(traj) >= 2:
            pts = np.array(
                [self.world_to_pixel(x, y) for (x, y) in traj],
                dtype=np.int32,
            )
            cv2.polylines(img, [pts], False, COLOR_TRAIL, 2, lineType=cv2.LINE_AA)

        # ---- 4. Named waypoints ----
        for name, wp in self.state.named_waypoints.items():
            if not wp:
                continue
            px = self.world_to_pixel(wp[0], wp[1])
            cv2.circle(img, px, 6, COLOR_NAMED_WAYPOINT, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, px, 9, COLOR_NAMED_WAYPOINT, 1, lineType=cv2.LINE_AA)
            cv2.putText(img, name, (px[0] + 8, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NAMED_WAYPOINT, 1, cv2.LINE_AA)

        # ---- 5. Detected objects ----
        rx, ry, ryaw = self.get_pose()
        for det in list(self.state.detected_objects):
            if det.position_3d is None:
                continue
            # det.position_3d is in CAMERA optical frame (X right, Y down, Z forward).
            # Convert to base_link (X forward = Z_optical, Y left = -X_optical),
            # then to world via the robot's yaw.
            x_cam, _y_cam, z_cam = det.position_3d
            x_base = z_cam
            y_base = -x_cam
            cy = math.cos(ryaw); sy = math.sin(ryaw)
            world_x = rx + cy * x_base - sy * y_base
            world_y = ry + sy * x_base + cy * y_base
            px = self.world_to_pixel(world_x, world_y)
            cv2.circle(img, px, 4, COLOR_OBJECT, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, px, 7, COLOR_OBJECT, 1, lineType=cv2.LINE_AA)
            cv2.putText(img, det.class_name, (px[0] + 8, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OBJECT, 1, cv2.LINE_AA)

        # ---- 6. Camera FOV cone (overlay) ----
        # Approximate horizontal FOV from the intrinsics; defaults to ~70° if unknown.
        intr = self.state.camera_intrinsics
        if intr.is_valid() and intr.fx > 0:
            half_fov = math.atan(intr.width / (2.0 * intr.fx))
        else:
            half_fov = math.radians(35.0)
        cone_range_m = self.max_range_m
        left_yaw = ryaw + half_fov
        right_yaw = ryaw - half_fov
        p_robot = self.world_to_pixel(rx, ry)
        p_left = self.world_to_pixel(rx + cone_range_m * math.cos(left_yaw),
                                     ry + cone_range_m * math.sin(left_yaw))
        p_right = self.world_to_pixel(rx + cone_range_m * math.cos(right_yaw),
                                      ry + cone_range_m * math.sin(right_yaw))
        # Filled cone with low alpha (overlay then blend)
        overlay = img.copy()
        cone_pts = np.array([p_robot, p_left, p_right], dtype=np.int32)
        cv2.fillConvexPoly(overlay, cone_pts, COLOR_FOV)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0.0, dst=img)
        cv2.line(img, p_robot, p_left, COLOR_FOV, 1, lineType=cv2.LINE_AA)
        cv2.line(img, p_robot, p_right, COLOR_FOV, 1, lineType=cv2.LINE_AA)

        # ---- 7. Robot body — filled circle + heading arrow + outline ring ----
        body_radius_m = 0.16          # QBot3 base diameter ~32 cm
        body_radius_px = max(6, int(body_radius_m / self.resolution))
        cv2.circle(img, p_robot, body_radius_px, COLOR_ROBOT_BODY, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, p_robot, body_radius_px + 1, COLOR_ROBOT, 2, lineType=cv2.LINE_AA)
        # Heading arrow extends 0.30 m forward
        head_x = rx + 0.30 * math.cos(ryaw)
        head_y = ry + 0.30 * math.sin(ryaw)
        p_head = self.world_to_pixel(head_x, head_y)
        cv2.arrowedLine(img, p_robot, p_head, COLOR_ROBOT, 3,
                        line_type=cv2.LINE_AA, tipLength=0.35)

        # ---- 8. Scale bar (1 m) and HUD ----
        bar_len_px = cells_per_meter
        cv2.rectangle(img, (8, h - 24), (8 + bar_len_px + 4, h - 6),
                      (0, 0, 0), -1)
        cv2.line(img, (10, h - 14), (10 + bar_len_px, h - 14),
                 COLOR_HUD_TEXT, 2)
        cv2.putText(img, "1 m", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        # Top-left HUD: pose + counts (so the user can see SLAM is alive)
        hud_lines = [
            f"x={rx:+.2f}  y={ry:+.2f}  yaw={math.degrees(ryaw):+.0f}",
            f"obstacles: {occ_count:>5}     trail: {len(traj):>4}",
            f"frames: int={self._frames_integrated:>4}  skip={self._frames_skipped_motion:>3}",
        ]
        for i, line in enumerate(hud_lines):
            y = 18 + i * 16
            cv2.putText(img, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        COLOR_HUD_DIM if i > 0 else COLOR_HUD_TEXT,
                        1, cv2.LINE_AA)

        # Cardinal compass (small, top-right)
        cx, cy = w - 28, 28
        cv2.circle(img, (cx, cy), 14, COLOR_HUD_DIM, 1, lineType=cv2.LINE_AA)
        cv2.putText(img, "N", (cx - 4, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, "E", (cx + 18, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_HUD_DIM, 1, cv2.LINE_AA)

        # ---- 9. Upscale for the GUI ----
        # The morphological cleanup gives us crisp 1-cell edges; LANCZOS4
        # preserves them when we resize to the viewer resolution. We do
        # the upscale here instead of leaving it to QPixmap.scaled() so
        # the trajectory polyline, text, and FOV cone are anti-aliased at
        # the final output size (drawing them on the small grid first and
        # then upscaling with NEAREST is what made the old renderer look
        # blocky).
        if RENDER_UPSCALE > 1:
            img = cv2.resize(
                img,
                (w * RENDER_UPSCALE, h * RENDER_UPSCALE),
                interpolation=cv2.INTER_LANCZOS4,
            )

        self._last_render = img
