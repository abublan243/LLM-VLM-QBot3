"""
Manual Mode — keyboard / D-pad teleop.

Maintains a 'pressed-keys' map updated by the GUI on key-down/up events
(and by the on-screen D-pad's pressed/released signals). A 10 Hz timer
turns that into a /qbot3/cmd_vel publish at the configured speed limits.

Also snapshots manual driving waypoints into SharedState every second so
the AI mode can use them as breadcrumbs in its LLM context.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.shared_state import SharedState

logger = logging.getLogger(__name__)


# Logical action keys (mapped by the GUI from WASD / arrows / D-pad buttons)
ACTIONS = ("forward", "backward", "left", "right")


class ManualMode(QObject):
    """Continuous-velocity teleop coordinator. Activate to enable publishing."""

    velocity_changed = pyqtSignal(float, float)   # (linear_x, angular_z)
    activated = pyqtSignal()
    deactivated = pyqtSignal()

    def __init__(
        self,
        state: SharedState,
        ros: Any,
        *,
        max_linear: float = 0.3,
        max_angular: float = 1.5,
        publish_hz: float = 10.0,
        waypoint_hz: float = 1.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)

        self._linear_scale = 1.0       # 0..1 (slider)
        self._angular_scale = 1.0      # 0..1 (slider)
        self._pressed = {a: False for a in ACTIONS}
        self._active = False

        self._publish_timer = QTimer(self)
        self._publish_timer.setInterval(int(1000 / max(1.0, publish_hz)))
        self._publish_timer.timeout.connect(self._publish_tick)

        self._waypoint_timer = QTimer(self)
        self._waypoint_timer.setInterval(int(1000 / max(0.2, waypoint_hz)))
        self._waypoint_timer.timeout.connect(self._waypoint_tick)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def activate(self) -> None:
        if self._active:
            return
        self._active = True
        for a in ACTIONS:
            self._pressed[a] = False
        self._publish_timer.start()
        self._waypoint_timer.start()
        self.state.set_active_mode("manual")
        self.activated.emit()
        logger.debug("Manual mode activated")

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        self._publish_timer.stop()
        self._waypoint_timer.stop()
        # One final zero command so the robot doesn't keep its last commanded velocity
        try:
            self.ros.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass
        self.deactivated.emit()
        logger.debug("Manual mode deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    def set_speed_limits(self, max_linear: float, max_angular: float) -> None:
        self.max_linear = max(0.0, float(max_linear))
        self.max_angular = max(0.0, float(max_angular))

    def set_linear_scale(self, scale_0_1: float) -> None:
        self._linear_scale = max(0.0, min(1.0, float(scale_0_1)))

    def set_angular_scale(self, scale_0_1: float) -> None:
        self._angular_scale = max(0.0, min(1.0, float(scale_0_1)))

    # ---------------------------------------------------------------
    # Input — called from the GUI
    # ---------------------------------------------------------------

    def press(self, action: str) -> None:
        if action in self._pressed:
            self._pressed[action] = True

    def release(self, action: str) -> None:
        if action in self._pressed:
            self._pressed[action] = False

    def stop(self) -> None:
        for a in ACTIONS:
            self._pressed[a] = False
        if self._active:
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass
            self.velocity_changed.emit(0.0, 0.0)

    def save_base(self) -> None:
        self.state.save_named_waypoint("base")

    # ---------------------------------------------------------------
    # Internal — publish + waypoint timers
    # ---------------------------------------------------------------

    def _publish_tick(self) -> None:
        if not self._active:
            return
        f = 1 if self._pressed["forward"] else 0
        b = 1 if self._pressed["backward"] else 0
        l = 1 if self._pressed["left"] else 0
        r = 1 if self._pressed["right"] else 0
        linear_x = (f - b) * self.max_linear * self._linear_scale
        angular_z = (l - r) * self.max_angular * self._angular_scale

        try:
            self.ros.publish_cmd_vel(linear_x, angular_z)
        except Exception as exc:
            logger.debug("manual cmd_vel publish skipped: %s", exc)
            return
        self.velocity_changed.emit(linear_x, angular_z)

    def _waypoint_tick(self) -> None:
        # Only record while the robot is moving — keeps the breadcrumb list tight
        with self.state.lock:
            v = abs(self.state.odom.linear_x) + abs(self.state.odom.angular_z)
        if v > 0.01:
            self.state.append_manual_waypoint()
