"""
SLAMViewerWidget — standalone SLAM map view with simple zoom + recentring.

Mostly used inside CameraViewerWidget's SLAM tab via _SLAMView, but exposed
as its own widget for any panel that wants a larger map. Adds:
    * Reset button (recentre on robot)
    * Reset map button (clears occupancy grid + trajectory)
    * Live coverage / nearest-obstacle readouts
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens
from gui.widgets.camera_viewer import AspectImageLabel


class SLAMViewerWidget(QWidget):
    def __init__(
        self,
        state: SharedState,
        slam_manager: Any,
        *,
        refresh_hz: float = 5.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.slam = slam_manager

        self._image = AspectImageLabel(self, placeholder="MAPPING…")

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self._reset_btn = QPushButton("Reset map")
        self._reset_btn.setProperty("variant", "ghost")
        self._reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(self._reset_btn)

        toolbar.addStretch(1)

        self._coverage_label = QLabel("")
        self._coverage_label.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; font-family: 'JetBrains Mono'; font-size: 11px;"
        )
        toolbar.addWidget(self._coverage_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._image, 1)
        layout.addLayout(toolbar)

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / max(1.0, refresh_hz)))
        self._timer.timeout.connect(self.refresh)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
            self.refresh()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def refresh(self) -> None:
        if self.slam is None:
            return
        try:
            img = self.slam.get_map_image()
            nearest = self.slam.get_nearest_obstacle_distance()
        except Exception:
            return
        if img is not None and img.size > 0:
            self._image.set_bgr(img)
        with self.state.lock:
            traj = len(self.state.slam_trajectory)
            wp = len(self.state.named_waypoints)
        nearest_txt = f"{nearest:.2f} m" if nearest != float("inf") else "—"
        self._coverage_label.setText(
            f"trajectory {traj}  ·  waypoints {wp}  ·  nearest {nearest_txt}"
        )

    def _on_reset(self) -> None:
        if self.slam is not None and hasattr(self.slam, "reset"):
            self.slam.reset()
