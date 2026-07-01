"""
PointcloudViewerWidget — accumulating world-frame 3D map with mouse control.

Each refresh tick:
    1. Reconstruct a point cloud from the latest depth + intrinsics.
    2. Transform optical → base_link (camera height + pitch correction so
       the floor reads horizontal).
    3. Apply the robot's odom pose (x, y, yaw) → WORLD frame.
    4. Append the new points into a persistent Open3D PointCloud.
    5. Voxel-downsample whenever the cloud exceeds the size budget so memory
       stays bounded as the robot drives around.
    6. Render offscreen and blit to the QLabel.

This makes it a real 3D map of the room, not just the current view. The
operator drags the cloud with the mouse to look around (left-drag = rotate,
right-drag = pan, scroll = zoom). Auto-rotate disables on first interaction.

Fast-turn skip: when |angular velocity Z| > threshold the integration is
paused (motion blur + odom-yaw lag during quick turns smear obstacles into
unreliable arcs). The accumulated cloud stays on screen.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.sensor_processor import reconstruct_point_cloud, transform_optical_to_base_link
from core.shared_state import SharedState
from gui.theme import Tokens

logger = logging.getLogger(__name__)


# =====================================================================
# Mouse-aware label — captures drag/scroll and forwards to a callback
# =====================================================================


class _PointcloudCanvas(QLabel):
    """QLabel that turns mouse drag + wheel + +/- keys into rotate / pan / zoom callbacks."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._dragging = False
        self._last_pos = None
        self._cb: Optional[Callable[[str, float, float], None]] = None

    def set_mouse_callback(self, cb: Callable[[str, float, float], None]) -> None:
        self._cb = cb

    def enterEvent(self, ev) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        # Grab focus on hover so +/- keys go to us, not the active mode panel.
        self.setFocus()
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self.unsetCursor()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self._dragging = True
        self._last_pos = ev.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.setFocus()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if not self._dragging or self._last_pos is None or self._cb is None:
            return
        pos = ev.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        # Right or middle button → pan; left button → rotate.
        buttons = ev.buttons()
        if buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            self._cb("pan", float(dx), float(dy))
        else:
            self._cb("rotate", float(dx), float(dy))

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._dragging = False
        self._last_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        # Double-click = reset view (fast path, no toolbar trip)
        if self._cb is not None:
            self._cb("reset", 0.0, 0.0)

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if self._cb is None:
            return
        # Prefer the high-resolution pixelDelta when available (trackpad);
        # fall back to angleDelta for traditional wheels (120 per notch).
        pix = ev.pixelDelta().y() if hasattr(ev, "pixelDelta") else 0
        ang = ev.angleDelta().y()
        # Normalise to "notches": one mouse-wheel click = 1.0
        if pix:
            notches = pix / 30.0
        else:
            notches = ang / 120.0
        if notches != 0.0:
            self._cb("zoom", 0.0, float(notches))
        ev.accept()

    def keyPressEvent(self, ev) -> None:
        if self._cb is None:
            super().keyPressEvent(ev)
            return
        key = ev.key()
        # +/- = zoom in/out; arrow keys = small rotate; R = reset
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):     # = and + are same key on US layout
            self._cb("zoom", 0.0, 1.0)
        elif key == Qt.Key.Key_Minus:
            self._cb("zoom", 0.0, -1.0)
        elif key == Qt.Key.Key_Left:
            self._cb("rotate", -20.0, 0.0)
        elif key == Qt.Key.Key_Right:
            self._cb("rotate", 20.0, 0.0)
        elif key == Qt.Key.Key_Up:
            self._cb("rotate", 0.0, -20.0)
        elif key == Qt.Key.Key_Down:
            self._cb("rotate", 0.0, 20.0)
        elif key == Qt.Key.Key_R:
            self._cb("reset", 0.0, 0.0)
        else:
            super().keyPressEvent(ev)


# =====================================================================
# PointcloudViewerWidget
# =====================================================================


class PointcloudViewerWidget(QWidget):
    """Accumulating world-frame 3D map with mouse control and a Reset Map button."""

    # Round 20: render at a substantially larger resolution so the
    # QPixmap.scaled() blit doesn't have to upscale much when the SLAM
    # tab fills a typical 1080p viewport — that upscale was the main
    # source of the "resolution becomes bad" complaint while the robot
    # was moving (sparse cloud + smooth-interpolated upscale = blur).
    RENDER_W = 1280
    RENDER_H = 800

    def __init__(
        self,
        state: SharedState,
        *,
        refresh_hz: float = 6.0,
        camera_height_m: float = 0.10,
        camera_pitch_deg: float = 0.0,
        # Round 20: motion-skip gate loosened. The whole point of the
        # accumulated 3D map is that the operator sees the world fill in
        # as the robot moves around the room. Skipping at 0.6 rad/s was
        # cutting integration any time the robot scanned its head, which
        # is exactly when fresh viewpoints become available. Real motion
        # blur only kicks in above ~1.2 rad/s on the QBot3.
        max_angular_velocity_radps: float = 1.2,
        # World-frame map sizing
        voxel_size_m: float = 0.04,         # 4 cm voxels
        max_points: int = 350_000,          # ~75% more points before downsample
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_rad = math.radians(float(camera_pitch_deg))
        self.max_angular_velocity_radps = float(max_angular_velocity_radps)
        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)

        self._vis: Any = None
        self._geom: Any = None
        self._open3d_ok = True
        self._auto_rotate = True
        self._accumulate = True
        self._frame_idx = 0
        self._motion_skip_count = 0
        self._user_view_dirty = False     # cleared by Reset View; set by mouse drag

        # ---- UI ----
        self._image = _PointcloudCanvas(self)
        self._image.setMinimumSize(self.RENDER_W, self.RENDER_H)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.setStyleSheet(
            f"background-color: #050507;"
            f"color: {Tokens.TEXT_MUTED};"
            f"border: 1px solid {Tokens.BORDER};"
            f"border-radius: {Tokens.RADIUS_LG}px;"
            "font-family: 'JetBrains Mono', monospace;"
            "letter-spacing: 1.5px;"
        )
        self._image.setText("WAITING FOR POINT CLOUD")
        self._image.set_mouse_callback(self._on_mouse)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self._toggle_accumulate = QPushButton("Accumulate: ON")
        self._toggle_accumulate.setProperty("variant", "ghost")
        self._toggle_accumulate.setCheckable(True)
        self._toggle_accumulate.setChecked(True)
        self._toggle_accumulate.toggled.connect(self._on_toggle_accumulate)
        self._toggle_accumulate.setToolTip(
            "When ON, points are accumulated into a persistent world-frame "
            "map as the robot moves. When OFF, the viewer shows only the "
            "current depth frame."
        )
        toolbar.addWidget(self._toggle_accumulate)

        self._toggle_rotate = QPushButton("Auto-rotate: ON")
        self._toggle_rotate.setProperty("variant", "ghost")
        self._toggle_rotate.setCheckable(True)
        self._toggle_rotate.setChecked(True)
        self._toggle_rotate.toggled.connect(self._on_toggle_rotate)
        toolbar.addWidget(self._toggle_rotate)

        self._reset_view_btn = QPushButton("Reset view")
        self._reset_view_btn.setProperty("variant", "ghost")
        self._reset_view_btn.setToolTip("Recenter on the robot's current pose.")
        self._reset_view_btn.clicked.connect(self._on_reset_view)
        toolbar.addWidget(self._reset_view_btn)

        self._clear_map_btn = QPushButton("Clear map")
        self._clear_map_btn.setProperty("variant", "ghost")
        self._clear_map_btn.setToolTip("Discard the accumulated 3D map and start over.")
        self._clear_map_btn.clicked.connect(self._on_clear_map)
        toolbar.addWidget(self._clear_map_btn)

        toolbar.addStretch(1)

        self._stats = QLabel("0 points")
        self._stats.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        toolbar.addWidget(self._stats)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._image, 1)
        layout.addLayout(toolbar)

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / max(1.0, refresh_hz)))
        self._timer.timeout.connect(self.refresh)

    # ---------------------------------------------------------------
    # Visibility gating
    # ---------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    # ---------------------------------------------------------------
    # Lazy Open3D init
    # ---------------------------------------------------------------

    def _ensure_visualizer(self) -> bool:
        if self._vis is not None:
            return True
        if not self._open3d_ok:
            return False
        try:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=self.RENDER_W, height=self.RENDER_H, visible=False)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.02, 0.02, 0.03])
            # Round 20: bump point_size 1.6 → 2.4 so a moderately dense
            # cloud still reads as a solid surface during motion. The old
            # 1.6 px dots disappeared into the dark background whenever
            # the cloud thinned out (auto-rotate angles, post-downsample),
            # which is what made motion look "low resolution".
            opt.point_size = 2.4
            self._vis = vis
            return True
        except Exception as exc:
            logger.warning("Open3D not available — 3D tab disabled: %s", exc)
            self._open3d_ok = False
            self._image.setText("OPEN3D NOT AVAILABLE\n\n"
                                "Install with: pip install open3d")
            return False

    # ---------------------------------------------------------------
    # Refresh loop
    # ---------------------------------------------------------------

    def refresh(self) -> None:
        with self.state.lock:
            depth = None if self.state.depth_frame is None else self.state.depth_frame.copy()
            rgb = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            intr = self.state.camera_intrinsics
            ang_z = abs(self.state.imu.angular_velocity[2])
            pose_x = self.state.odom.x
            pose_y = self.state.odom.y
            pose_yaw = self.state.odom.yaw_rad

        if depth is None or not intr.is_valid():
            return
        if not self._ensure_visualizer():
            return

        is_turning = ang_z > self.max_angular_velocity_radps
        if is_turning:
            self._motion_skip_count += 1

        # Round 20: integrate every 2nd refresh instead of every 3rd
        # (~3 Hz integration at 6 Hz refresh). The cloud now grows fast
        # enough during motion that it stays visually dense across
        # viewpoint changes — the previous 2 Hz rate let the cloud
        # "thin out" relative to what auto-rotate was showing.
        if (not is_turning) and (self._frame_idx % 2 == 0):
            arr = reconstruct_point_cloud(depth, intr, rgb_bgr=rgb,
                                          stride=4, max_distance_m=4.0)
            if arr is not None and arr.size > 0:
                arr_base = transform_optical_to_base_link(
                    arr,
                    camera_height_m=self.camera_height_m,
                    camera_pitch_rad=self.camera_pitch_rad,
                )
                if self._accumulate:
                    self._accumulate_world(arr_base, pose_x, pose_y, pose_yaw)
                else:
                    self._replace_geometry(arr_base, pose_x, pose_y, pose_yaw)
                self._update_stats()
        self._frame_idx += 1

        # Turntable rotation only if the user hasn't taken control
        if self._auto_rotate and not self._user_view_dirty:
            ctrl = self._vis.get_view_control()
            ctrl.rotate(2.0, 0.0)

        # Render and blit
        try:
            self._vis.poll_events()
            self._vis.update_renderer()
            buf = self._vis.capture_screen_float_buffer(do_render=True)
            arr8 = (np.asarray(buf) * 255).clip(0, 255).astype(np.uint8)
        except Exception as exc:
            logger.debug("Open3D render skipped: %s", exc)
            return

        h, w, _ = arr8.shape
        rgb_arr = np.ascontiguousarray(arr8)
        img = QImage(rgb_arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._image.setPixmap(QPixmap.fromImage(img).scaled(
            self._image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    # ---------------------------------------------------------------
    # Geometry update — accumulating vs replacing
    # ---------------------------------------------------------------

    def _accumulate_world(self, base_xyzrgb: np.ndarray,
                          pose_x: float, pose_y: float, pose_yaw: float) -> None:
        """Append new points (transformed into world frame) to the persistent cloud."""
        try:
            import open3d as o3d
        except Exception:
            return

        new_xyz, new_rgb = self._to_world(base_xyzrgb, pose_x, pose_y, pose_yaw)
        if new_xyz.size == 0:
            return

        if self._geom is None:
            self._geom = o3d.geometry.PointCloud()
            self._geom.points = o3d.utility.Vector3dVector(new_xyz)
            if new_rgb is not None:
                self._geom.colors = o3d.utility.Vector3dVector(new_rgb)
            self._vis.add_geometry(self._geom)
            self._on_reset_view()
            return

        existing = np.asarray(self._geom.points)
        combined = np.concatenate([existing, new_xyz], axis=0)
        combined_c = None
        if new_rgb is not None and self._geom.has_colors():
            existing_c = np.asarray(self._geom.colors)
            combined_c = np.concatenate([existing_c, new_rgb], axis=0)

        # Voxel-downsample once we exceed the budget — keeps memory bounded
        # and visually thins out duplicates from re-observing the same region.
        if len(combined) > self.max_points:
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(combined)
            if combined_c is not None:
                tmp.colors = o3d.utility.Vector3dVector(combined_c)
            ds = tmp.voxel_down_sample(self.voxel_size_m)
            self._geom.points = ds.points
            if ds.has_colors():
                self._geom.colors = ds.colors
        else:
            self._geom.points = o3d.utility.Vector3dVector(combined)
            if combined_c is not None:
                self._geom.colors = o3d.utility.Vector3dVector(combined_c)

        self._vis.update_geometry(self._geom)

    def _replace_geometry(self, base_xyzrgb: np.ndarray,
                          pose_x: float, pose_y: float, pose_yaw: float) -> None:
        """Show only the current frame (no accumulation) — same world transform."""
        try:
            import open3d as o3d
        except Exception:
            return
        new_xyz, new_rgb = self._to_world(base_xyzrgb, pose_x, pose_y, pose_yaw)
        if new_xyz.size == 0:
            return
        if self._geom is None:
            self._geom = o3d.geometry.PointCloud()
            self._geom.points = o3d.utility.Vector3dVector(new_xyz)
            if new_rgb is not None:
                self._geom.colors = o3d.utility.Vector3dVector(new_rgb)
            self._vis.add_geometry(self._geom)
            self._on_reset_view()
            return
        self._geom.points = o3d.utility.Vector3dVector(new_xyz)
        if new_rgb is not None:
            self._geom.colors = o3d.utility.Vector3dVector(new_rgb)
        self._vis.update_geometry(self._geom)

    def _to_world(self, base_xyzrgb: np.ndarray,
                  pose_x: float, pose_y: float, pose_yaw: float):
        """base_link XYZ → world XYZ via current odom pose."""
        if base_xyzrgb is None or base_xyzrgb.size == 0:
            return np.zeros((0, 3), dtype=np.float64), None
        cy = math.cos(pose_yaw)
        sy = math.sin(pose_yaw)
        bx = base_xyzrgb[:, 0]
        by = base_xyzrgb[:, 1]
        bz = base_xyzrgb[:, 2]
        wx = pose_x + cy * bx - sy * by
        wy = pose_y + sy * bx + cy * by
        wz = bz
        xyz = np.stack([wx, wy, wz], axis=1).astype(np.float64)
        rgb = None
        if base_xyzrgb.shape[1] >= 6:
            rgb = base_xyzrgb[:, 3:6].astype(np.float64)
        return xyz, rgb

    def _update_stats(self) -> None:
        if self._geom is None:
            self._stats.setText("0 points")
            return
        n = len(self._geom.points)
        bits = [f"{n:,} points"]
        if self._accumulate:
            bits.append("world map")
        else:
            bits.append("live frame")
        if self._motion_skip_count > 0 and self._frame_idx % 32 == 0:
            bits.append(f"{self._motion_skip_count} turns skipped")
        self._stats.setText("  ·  ".join(bits))

    # ---------------------------------------------------------------
    # Mouse callback (rotate / pan / zoom on the Open3D ViewControl)
    # ---------------------------------------------------------------

    def _on_mouse(self, kind: str, dx: float, dy: float) -> None:
        if self._vis is None:
            return
        # Reset is a fast-path for double-click / R key
        if kind == "reset":
            self._on_reset_view()
            return
        # First user interaction silences the auto-rotate so it doesn't
        # fight the user's hand.
        if self._auto_rotate and kind in ("rotate", "pan", "zoom"):
            self._auto_rotate = False
            self._toggle_rotate.blockSignals(True)
            self._toggle_rotate.setChecked(False)
            self._toggle_rotate.blockSignals(False)
            self._toggle_rotate.setText("Auto-rotate: OFF")
        self._user_view_dirty = True
        try:
            ctrl = self._vis.get_view_control()
            if kind == "rotate":
                ctrl.rotate(dx, dy)
            elif kind == "pan":
                # Open3D translate units are already in pixels for the offscreen
                # canvas; flip Y so dragging up pans the view up.
                ctrl.translate(dx, -dy)
            elif kind == "zoom":
                # `dy` is wheel notches (positive = scroll up).
                # Open3D's ViewControl.scale(s) ADDS s to the internal zoom
                # value. Smaller zoom value = camera closer = zoomed IN.
                # So scroll-up = zoom-in = negative scale.
                step = 0.35
                ctrl.scale(-step * dy)
        except Exception as exc:
            logger.debug("View-control gesture skipped: %s", exc)

    # ---------------------------------------------------------------
    # Toolbar handlers
    # ---------------------------------------------------------------

    def _on_toggle_accumulate(self, on: bool) -> None:
        self._accumulate = on
        self._toggle_accumulate.setText(f"Accumulate: {'ON' if on else 'OFF'}")
        if not on:
            # Switching to live-frame mode — ditch the accumulated map
            self._on_clear_map()

    def _on_toggle_rotate(self, on: bool) -> None:
        self._auto_rotate = on
        self._toggle_rotate.setText(f"Auto-rotate: {'ON' if on else 'OFF'}")
        if on:
            self._user_view_dirty = False

    def _on_clear_map(self) -> None:
        if self._vis is None or self._geom is None:
            return
        try:
            self._vis.remove_geometry(self._geom, reset_bounding_box=False)
        except Exception:
            pass
        self._geom = None
        self._motion_skip_count = 0
        self._stats.setText("0 points  ·  cleared")

    def _on_reset_view(self) -> None:
        """Frame the view so the robot's current pose is centred and the
        floor reads horizontal. World frame: X forward, Y left, Z up.
        """
        if self._vis is None:
            return
        with self.state.lock:
            rx = self.state.odom.x
            ry = self.state.odom.y
        try:
            ctrl = self._vis.get_view_control()
            ctrl.set_zoom(0.45)
            ctrl.set_lookat([rx + 0.5, ry, 0.30])     # follow the robot
            ctrl.set_front([-0.55, -0.30, 0.78])      # camera back-left-up
            ctrl.set_up([0.0, 0.0, 1.0])
        except Exception as exc:
            logger.debug("Reset view skipped: %s", exc)
        self._user_view_dirty = False

    def closeEvent(self, event) -> None:
        try:
            if self._vis is not None:
                self._vis.destroy_window()
                self._vis = None
        except Exception:
            pass
        super().closeEvent(event)
