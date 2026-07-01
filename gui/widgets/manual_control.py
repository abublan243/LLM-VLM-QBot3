"""
ManualControlWidget — D-pad teleop + speed sliders + live odometry readout.

Hooks into ModeManual via press(action) / release(action). The MainWindow
forwards keyboard W/A/S/D and arrow keys directly to ModeManual; this
widget covers the on-screen control path.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens


class ManualControlWidget(QWidget):
    """Mode-2 right panel: drive the robot with on-screen buttons + sliders."""

    save_base_clicked = pyqtSignal()

    # Slider ranges — match config/settings.json safety caps
    LINEAR_MAX_MPS = 0.5
    ANGULAR_MAX_RADPS = 1.5

    def __init__(
        self,
        state: SharedState,
        manual_mode,                    # ModeManual instance (avoid Qt import cycle)
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._mm = manual_mode

        self._build()
        self._wire()

        # Pose readout refreshes from SharedState (ROS bridge writes it on each odom msg).
        # Paused via showEvent/hideEvent so we don't poll when the user is in
        # AI / Skills mode and this panel is in a hidden QStackedWidget slot.
        self._readout_timer = QTimer(self)
        self._readout_timer.setInterval(100)
        self._readout_timer.timeout.connect(self._refresh_readouts)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._readout_timer.isActive():
            self._readout_timer.start()
            self._refresh_readouts()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._readout_timer.stop()

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        outer.addWidget(self._build_dpad_card())
        outer.addWidget(self._build_speed_card())
        outer.addWidget(self._build_readout_card())

        save_btn = QPushButton("Save current pose as 'base'")
        save_btn.setProperty("variant", "secondary")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self.save_base_clicked.emit)
        outer.addWidget(save_btn)

        outer.addStretch(1)

    def _build_dpad_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        cap = QLabel("DRIVE")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        hint = QLabel("Press buttons or use W A S D / arrow keys")
        hint.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 11px;")
        v.addWidget(hint)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 8, 0, 0)

        self._btn_fwd   = self._dpad_button("▲", "forward")    # ▲
        self._btn_back  = self._dpad_button("▼", "backward")   # ▼
        self._btn_left  = self._dpad_button("◀", "left")       # ◀
        self._btn_right = self._dpad_button("▶", "right")      # ▶

        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setProperty("variant", "danger")
        self._btn_stop.setMinimumHeight(56)
        self._btn_stop.clicked.connect(self._mm.stop)

        grid.addWidget(self._btn_fwd,   0, 1)
        grid.addWidget(self._btn_left,  1, 0)
        grid.addWidget(self._btn_stop,  1, 1)
        grid.addWidget(self._btn_right, 1, 2)
        grid.addWidget(self._btn_back,  2, 1)
        for col in range(3):
            grid.setColumnStretch(col, 1)
        v.addLayout(grid)
        return card

    def _dpad_button(self, label: str, action: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setProperty("variant", "dpad")
        btn.setAutoRepeat(False)
        btn.pressed.connect(lambda a=action: self._mm.press(a))
        btn.released.connect(lambda a=action: self._mm.release(a))
        return btn

    def _build_speed_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        cap = QLabel("SPEED")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        # Defaults: 0.30 m/s linear, 1.00 rad/s angular (60% / 67% of max)
        self._linear_slider, self._linear_value = self._slider_row(
            "Linear (m/s)", 0.0, self.LINEAR_MAX_MPS, 0.30, self._on_linear,
        )
        v.addLayout(self._linear_slider)
        self._angular_slider, self._angular_value = self._slider_row(
            "Angular (rad/s)", 0.0, self.ANGULAR_MAX_RADPS, 1.00, self._on_angular,
        )
        v.addLayout(self._angular_slider)
        return card

    def _slider_row(self, label: str, lo: float, hi: float, default: float,
                    handler) -> tuple:
        h = QHBoxLayout()
        h.setSpacing(10)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        lbl.setMinimumWidth(108)
        h.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        if hi > lo:
            slider.setValue(int((default - lo) / (hi - lo) * 100))
        h.addWidget(slider, 1)

        readout = QLabel(f"{default:.2f}")
        readout.setStyleSheet(
            f"color: {Tokens.ACCENT_SECONDARY};"
            "font-family: 'JetBrains Mono', monospace;"
            "font-size: 12px; min-width: 56px;"
        )
        readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(readout)

        def _on_value(v: int) -> None:
            value = lo + (hi - lo) * v / 100.0
            readout.setText(f"{value:.2f}")
            handler(value)

        slider.valueChanged.connect(_on_value)
        # Push initial value through so handler reflects the default
        handler(default)
        return h, readout

    def _build_readout_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        cap = QLabel("LIVE")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        self._vel_label = QLabel("v=+0.00 m/s   ω=+0.00 rad/s")     # ω
        self._vel_label.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY};"
            "font-family: 'JetBrains Mono', monospace;"
            "font-size: 13px;"
        )
        v.addWidget(self._vel_label)

        self._pose_label = QLabel("x=+0.00  y=+0.00  θ=+0°")    # θ °
        self._pose_label.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY};"
            "font-family: 'JetBrains Mono', monospace;"
            "font-size: 12px;"
        )
        v.addWidget(self._pose_label)
        return card

    # ---------------------------------------------------------------
    # Wiring + handlers
    # ---------------------------------------------------------------

    def _wire(self) -> None:
        if self._mm is not None:
            self._mm.velocity_changed.connect(self._on_velocity)

    def _on_linear(self, value: float) -> None:
        # ModeManual interprets `linear_x = sign * max_linear * scale`,
        # so we map the absolute slider value to a normalized scale.
        if self._mm is None:
            return
        if self._mm.max_linear > 0:
            self._mm.set_linear_scale(min(1.0, value / self._mm.max_linear))

    def _on_angular(self, value: float) -> None:
        if self._mm is None:
            return
        if self._mm.max_angular > 0:
            self._mm.set_angular_scale(min(1.0, value / self._mm.max_angular))

    def _on_velocity(self, lin: float, ang: float) -> None:
        self._vel_label.setText(
            f"v={lin:+.2f} m/s   ω={ang:+.2f} rad/s"
        )

    def _refresh_readouts(self) -> None:
        with self.state.lock:
            o = self.state.odom
        self._pose_label.setText(
            f"x={o.x:+.2f}  y={o.y:+.2f}  θ={math.degrees(o.yaw_rad):+.0f}°"
        )

    # ---------------------------------------------------------------
    # Keyboard helper for MainWindow
    # ---------------------------------------------------------------

    @staticmethod
    def keymap() -> dict:
        """Map Qt.Key enum values to ModeManual action names."""
        return {
            Qt.Key.Key_W: "forward",
            Qt.Key.Key_S: "backward",
            Qt.Key.Key_A: "left",
            Qt.Key.Key_D: "right",
            Qt.Key.Key_Up: "forward",
            Qt.Key.Key_Down: "backward",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
        }
