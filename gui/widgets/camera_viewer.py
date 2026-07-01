"""
CameraViewerWidget — the upper visualization slot.

Four tabs:
    1. RGB    — live colour feed with YOLO bounding-box overlay
    2. Depth  — colorized depth heatmap (uses /camera/depth/visual from the Pi)
    3. SLAM   — 2D occupancy grid from SlamManager
    4. 3D     — Open3D point cloud (offscreen-rendered, blitted into a QLabel)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens

logger = logging.getLogger(__name__)


# =====================================================================
# Generic AspectImage label — keeps aspect ratio while filling its slot
# =====================================================================


class AspectImageLabel(QLabel):
    """QLabel subclass that paints a numpy BGR / BGRA image scaled to fit."""

    def __init__(self, parent: Optional[QWidget] = None, *, placeholder: str = "no signal") -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background-color: {Tokens.BG};"
            f"color: {Tokens.TEXT_MUTED};"
            f"border: 1px solid {Tokens.BORDER};"
            f"border-radius: {Tokens.RADIUS_LG}px;"
            "font-family: 'JetBrains Mono', monospace;"
            "letter-spacing: 1.5px;"
        )
        self._placeholder = placeholder
        self._pixmap: Optional[QPixmap] = None
        self.setText(placeholder)

    def set_bgr(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            self.clear_image()
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        self._refresh_pixmap()

    def clear_image(self) -> None:
        self._pixmap = None
        self.setText(self._placeholder)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


# =====================================================================
# RGB tab — overlays YOLO detections on the colour frame
# =====================================================================


class _RGBView(QWidget):
    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._image = AspectImageLabel(self, placeholder="WAITING FOR CAMERA")
        self._stats = QLabel("0 detections")
        self._stats.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; font-family: 'JetBrains Mono'; font-size: 11px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._image, 1)
        layout.addWidget(self._stats)

    def refresh(self) -> None:
        with self.state.lock:
            frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            detections = list(self.state.detected_objects)
        if frame is None:
            self._image.clear_image()
            return
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 99, 108), 2)   # BGR of accent purple
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.distance_m and det.distance_m > 0:
                label += f"  {det.distance_m:.2f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1),
                          (255, 99, 108), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 245), 1, cv2.LINE_AA)
        self._image.set_bgr(frame)
        self._stats.setText(f"{len(detections)} detection{'s' if len(detections) != 1 else ''}")


# =====================================================================
# Depth tab — uses Pi-side colorized heatmap, with object distance overlays
# =====================================================================


class _DepthView(QWidget):
    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._image = AspectImageLabel(self, placeholder="WAITING FOR DEPTH")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image, 1)

    def refresh(self) -> None:
        with self.state.lock:
            visual = None if self.state.depth_visual_frame is None else self.state.depth_visual_frame.copy()
            depth = None if self.state.depth_frame is None else self.state.depth_frame.copy()
            detections = list(self.state.detected_objects)

        if visual is None and depth is not None:
            # Fall back to host-side colorize if the Pi visual stream isn't there
            clipped = np.clip(depth, 0, 3000).astype(np.float32) / 3000.0
            inv = (255 - clipped * 255).astype(np.uint8)
            visual = cv2.applyColorMap(inv, cv2.COLORMAP_JET)
            mask = depth == 0
            visual[mask] = (0, 0, 0)

        if visual is None:
            self._image.clear_image()
            return

        for det in detections:
            cx, cy = det.centroid_xy
            cv2.circle(visual, (cx, cy), 5, (255, 255, 255), 1)
            if det.distance_m > 0:
                txt = f"{det.distance_m:.2f} m"
                cv2.putText(visual, txt, (cx + 8, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
        self._image.set_bgr(visual)


# =====================================================================
# SLAM tab — host-side log-odds occupancy grid render
# =====================================================================


class _SLAMView(QWidget):
    def __init__(self, slam_manager: Any, ros_bridge: Any = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._slam = slam_manager
        self._ros = ros_bridge
        self._image = AspectImageLabel(self, placeholder="MAPPING…")

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addStretch(1)

        self._reset_btn = QPushButton("Reset map + odom + gyro")
        self._reset_btn.setProperty("variant", "ghost")
        self._reset_btn.setToolTip(
            "One click does everything:\n"
            "  • clears the host-side occupancy grid + trajectory\n"
            "  • zeroes the Pi's odometry pose\n"
            "  • re-triggers the 5 s gyro-bias calibration\n"
            "Use this whenever the map looks wrong — broken encoder spike, "
            "yaw drift, or you've physically moved the robot to a new starting spot."
        )
        self._reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(self._reset_btn)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 10px;"
        )
        toolbar.addWidget(self._status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._image, 1)
        layout.addLayout(toolbar)

    def refresh(self) -> None:
        if self._slam is None:
            self._image.clear_image()
            return
        try:
            img = self._slam.get_map_image()
        except Exception as exc:
            logger.exception("SLAM render failed: %s", exc)
            return
        if img is None or img.size == 0:
            self._image.clear_image()
            return
        self._image.set_bgr(img)

    def _on_reset(self) -> None:
        notes = []
        if self._slam is not None and hasattr(self._slam, "reset"):
            try:
                self._slam.reset()
                notes.append("grid cleared")
            except Exception as exc:
                logger.exception("SLAM reset failed: %s", exc)
                notes.append("grid reset FAILED")
        if self._ros is not None and hasattr(self._ros, "publish_reset_odom"):
            try:
                ok = self._ros.publish_reset_odom()
                notes.append("odom zeroed" if ok else "odom NOT reset (bridge?)")
            except Exception as exc:
                logger.exception("Pi odom reset failed: %s", exc)
                notes.append("odom FAILED")
        if self._ros is not None and hasattr(self._ros, "publish_reset_yaw"):
            try:
                ok = self._ros.publish_reset_yaw()
                notes.append("gyro recal triggered (5 s)" if ok else "gyro recal NOT sent")
            except Exception as exc:
                logger.exception("Yaw reset failed: %s", exc)
                notes.append("gyro recal FAILED")
        self._status.setText("  ·  ".join(notes) if notes else "no reset target wired")


# =====================================================================
# 3D Point Cloud tab — uses pointcloud_viewer subwidget
# =====================================================================


from gui.widgets.pointcloud_viewer import PointcloudViewerWidget   # noqa: E402  (after class defs)


# =====================================================================
# CameraViewerWidget — public top-level
# =====================================================================


class CameraViewerWidget(QWidget):
    """Tabbed visualization widget. Drives its own ~30 Hz refresh timer."""

    def __init__(
        self,
        state: SharedState,
        ros_bridge: Any,
        slam_manager: Any,
        *,
        display_fps: int = 30,
        camera_height_m: float = 0.10,
        camera_pitch_deg: float = 0.0,
        max_angular_velocity_radps: float = 0.6,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros_bridge = ros_bridge

        self._tabs = QTabWidget(self)
        # Live = pure RGB, event-driven on rgb_updated signal, no overlays.
        # Lives next to the YOLO-overlaid RGB sub-tab so the operator can
        # flip between detector view and clean view without leaving Camera.
        from gui.widgets.live_rgb_widget import LiveRGBWidget
        self._live = LiveRGBWidget(state, ros_bridge)
        self._rgb = _RGBView(state)
        self._depth = _DepthView(state)
        self._slam = _SLAMView(slam_manager, ros_bridge=ros_bridge)
        self._pcd = PointcloudViewerWidget(
            state,
            camera_height_m=camera_height_m,
            camera_pitch_deg=camera_pitch_deg,
            max_angular_velocity_radps=max_angular_velocity_radps,
        )
        self._tabs.addTab(self._live, "Live")
        self._tabs.addTab(self._rgb, "RGB")
        self._tabs.addTab(self._depth, "Depth")
        self._tabs.addTab(self._slam, "SLAM")
        self._tabs.addTab(self._pcd, "3D")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._tabs)

        # Cap display FPS to 20 by default — 30 Hz redraws of 640x480 BGR
        # frames consume ~28 MB/s of memory bandwidth and the human eye
        # doesn't notice the difference on a sensor stream.
        capped_fps = max(1, min(int(display_fps), 20))
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(int(1000 / capped_fps))
        self._refresh_timer.timeout.connect(self._refresh_active)

        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ---------------------------------------------------------------
    # Pause refresh when this widget isn't visible — the QTabWidget that
    # hosts us in MainWindow fires hideEvent on tab switch, so this saves
    # the ~30 Hz BGR copy/scale loop whenever the user is on Sensors / AI
    # Thought / Performance tabs.

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()
            # Repaint immediately so the user doesn't see stale content
            self._refresh_active()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._refresh_timer.stop()

    def _refresh_active(self) -> None:
        idx = self._tabs.currentIndex()
        # idx 0 = Live — event-driven on rgb_updated, no poll needed.
        if idx == 1:
            self._rgb.refresh()
        elif idx == 2:
            self._depth.refresh()
        elif idx == 3:
            self._slam.refresh()
        elif idx == 4:
            self._pcd.refresh()

    def _on_tab_changed(self, _idx: int) -> None:
        # Prompt an immediate refresh so the new tab doesn't show stale content
        self._refresh_active()
