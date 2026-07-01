"""
LiveRGBWidget — bare-bones, high-FPS RGB-only viewer.

Why this exists separately from VisionWidget / CameraViewerWidget:
    * No YOLO bounding-box compositing on every frame
    * No detection list rebuild
    * No 20 Hz poll timer — paint is driven by the bridge's `rgb_updated`
      signal, so we paint exactly when a new frame arrives upstream
      (whatever the Pi delivers, ~15 Hz nominally).

The widget is intentionally minimal: one full-bleed image label + a small
FPS counter overlay. When the tab is hidden, the slot short-circuits so
we don't waste CPU painting an off-screen widget.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.shared_state import SharedState
from gui.theme import Tokens
from gui.widgets.camera_viewer import AspectImageLabel

logger = logging.getLogger(__name__)


class LiveRGBWidget(QWidget):
    """Pure RGB stream — event-driven, no overlays, no compositing."""

    def __init__(
        self,
        state: SharedState,
        ros_bridge: Any,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros_bridge

        self._frame_ts: Deque[float] = deque(maxlen=60)
        self._is_visible = False

        self._build()

        # Event-driven repaint: hook the bridge's per-frame signal directly.
        # Using a queued connection (auto across threads) means each frame
        # arriving on the rclpy executor thread is delivered to the GUI
        # thread for painting without us doing any polling.
        if hasattr(ros_bridge, "rgb_updated"):
            ros_bridge.rgb_updated.connect(self._on_rgb_updated)

    # ---------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._image = AspectImageLabel(self, placeholder="LIVE RGB — waiting for frames")
        layout.addWidget(self._image, 1)

        # Tiny status strip at the bottom — display FPS + frame count
        self._status = QLabel("FPS: --   ·   frames: 0", self)
        self._status.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px; "
            f"padding: 2px 6px;"
        )
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._status)

    # ---------------------------------------------------------------
    # Show / hide
    # ---------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._is_visible = True
        # Re-paint the latest frame immediately on tab activation
        self._on_rgb_updated()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._is_visible = False
        self._frame_ts.clear()

    # ---------------------------------------------------------------
    # Frame slot
    # ---------------------------------------------------------------

    def _on_rgb_updated(self) -> None:
        if not self._is_visible:
            return
        frame = None
        with self.state.lock:
            if self.state.rgb_frame is not None:
                frame = self.state.rgb_frame
        if frame is None:
            return
        self._image.set_bgr(frame)

        now = time.monotonic()
        self._frame_ts.append(now)
        # Sliding-window FPS over the last second of timestamps
        cutoff = now - 1.0
        recent = [t for t in self._frame_ts if t >= cutoff]
        fps = len(recent)  # frames in the last 1 s
        h, w = frame.shape[:2]
        self._status.setText(
            f"FPS: {fps:>2}   ·   {w}×{h}   ·   frames: {len(self._frame_ts)}"
        )
