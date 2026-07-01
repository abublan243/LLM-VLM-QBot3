"""
SensorPanelWidget — pyqtgraph-based live telemetry dashboard.

Six readouts in a 2-column grid:
    * IMU accel (3-axis line plot)
    * IMU gyro  (3-axis line plot)
    * Encoder velocity (left vs right line plot)
    * Battery   (large numeric + voltage line + thin progress bar)
    * Cliff     (3 horizontal LED bars)
    * Bump      (3 LED dots that flash red on trigger)

Updates from ROSBridge Qt signals — no SharedState polling.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from core.shared_state import SharedState
from gui.theme import Tokens


# =====================================================================
# pyqtgraph defaults — apply once on first plot creation
# =====================================================================

pg.setConfigOptions(antialias=True, useOpenGL=False)
pg.setConfigOption("background", Tokens.SURFACE)
pg.setConfigOption("foreground", Tokens.TEXT_SECONDARY)


def _make_plot(title: str, *, ylabel: str = "") -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setTitle(title, color=Tokens.TEXT_SECONDARY, size="10pt")
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    if ylabel:
        plot.setLabel("left", ylabel, color=Tokens.TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(QColor(Tokens.TEXT_MUTED))
    plot.getAxis("left").setTextPen(QColor(Tokens.TEXT_MUTED))
    plot.getAxis("bottom").setPen(QColor(Tokens.BORDER))
    plot.getAxis("left").setPen(QColor(Tokens.BORDER))
    plot.getViewBox().setBackgroundColor(QColor(Tokens.SURFACE))
    return plot


# =====================================================================
# LED-style indicator
# =====================================================================


class _LEDIndicator(QLabel):
    """Small circular indicator. tone in {ok, warn, bad, info, dim}."""

    SIZE = 18

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setProperty("role", "led")
        self._label = label
        self.set_active(False)

    def set_active(self, active: bool, tone: str = "bad") -> None:
        color = {
            "ok": Tokens.SUCCESS,
            "warn": Tokens.WARNING,
            "bad": Tokens.DANGER,
            "info": Tokens.ACCENT_SECONDARY,
        }.get(tone, Tokens.DANGER)
        if active:
            self.setStyleSheet(
                f"background-color: {color};"
                f"border-radius: {self.SIZE//2}px;"
                f"border: 2px solid rgba(255,255,255,30);"
            )
        else:
            self.setStyleSheet(
                f"background-color: {Tokens.SURFACE_ELEVATED};"
                f"border-radius: {self.SIZE//2}px;"
                f"border: 1px solid {Tokens.BORDER};"
            )


# =====================================================================
# SensorPanelWidget
# =====================================================================


class SensorPanelWidget(QWidget):
    HISTORY = 200    # samples per plot (~10 seconds at 20 Hz tick)

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

        # ---- state buffers ----
        self._buf_accel = [deque(maxlen=self.HISTORY) for _ in range(3)]
        self._buf_gyro  = [deque(maxlen=self.HISTORY) for _ in range(3)]
        self._buf_vel_l: Deque[float] = deque(maxlen=self.HISTORY)
        self._buf_vel_r: Deque[float] = deque(maxlen=self.HISTORY)
        self._buf_battery_v: Deque[float] = deque(maxlen=self.HISTORY)
        self._buf_x = deque(maxlen=self.HISTORY)
        self._t = 0
        self._last_enc_l: Optional[int] = None
        self._last_enc_r: Optional[int] = None
        self._last_enc_t: float = 0.0

        # ---- plots ----
        self._plot_accel = _make_plot("Acceleration (m/s²)")
        self._curve_ax = self._plot_accel.plot(pen=pg.mkPen(Tokens.DANGER, width=1.5), name="x")
        self._curve_ay = self._plot_accel.plot(pen=pg.mkPen(Tokens.SUCCESS, width=1.5), name="y")
        self._curve_az = self._plot_accel.plot(pen=pg.mkPen(Tokens.ACCENT_SECONDARY, width=1.5), name="z")

        self._plot_gyro = _make_plot("Angular Velocity (rad/s)")
        self._curve_gx = self._plot_gyro.plot(pen=pg.mkPen(Tokens.DANGER, width=1.5))
        self._curve_gy = self._plot_gyro.plot(pen=pg.mkPen(Tokens.SUCCESS, width=1.5))
        self._curve_gz = self._plot_gyro.plot(pen=pg.mkPen(Tokens.ACCENT_SECONDARY, width=1.5))

        self._plot_enc = _make_plot("Wheel Velocity (m/s)")
        self._curve_vl = self._plot_enc.plot(pen=pg.mkPen(Tokens.ACCENT_PRIMARY, width=1.8), name="L")
        self._curve_vr = self._plot_enc.plot(pen=pg.mkPen(Tokens.ACCENT_SECONDARY, width=1.8), name="R")

        self._plot_batt = _make_plot("Battery (V)")
        self._curve_bv = self._plot_batt.plot(pen=pg.mkPen(Tokens.WARNING, width=1.8))

        # ---- numeric readouts ----
        self._battery_pct = QLabel("--%")
        self._battery_pct.setProperty("role", "display")
        self._battery_volts = QLabel("0.00 V")
        self._battery_volts.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-family: 'JetBrains Mono';")

        # ---- safety LEDs ----
        self._bump_leds = [_LEDIndicator("L"), _LEDIndicator("C"), _LEDIndicator("R")]
        self._cliff_leds = [_LEDIndicator("L"), _LEDIndicator("C"), _LEDIndicator("R")]

        self._build_layout()
        self._wire_signals()

        # Smoothed FPS-style tick to redraw plots — avoids re-pen on every msg.
        # Paused via showEvent/hideEvent so the plots only redraw when visible.
        self._draw_timer = QTimer(self)
        self._draw_timer.setInterval(100)   # 10 Hz redraw
        self._draw_timer.timeout.connect(self._redraw)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._draw_timer.isActive():
            self._draw_timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._draw_timer.stop()

    # ---------------------------------------------------------------

    def _build_layout(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        grid.addWidget(self._wrap("ACCELEROMETER", self._plot_accel), 0, 0)
        grid.addWidget(self._wrap("GYROSCOPE", self._plot_gyro), 0, 1)
        grid.addWidget(self._wrap("WHEEL VELOCITY", self._plot_enc), 1, 0)
        grid.addWidget(self._wrap_battery(), 1, 1)
        grid.addWidget(self._wrap_bumpcliff(), 2, 0, 1, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)

    def _wrap(self, title: str, content: QWidget) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        cap = QLabel(title)
        cap.setProperty("role", "caption")
        v.addWidget(cap)
        v.addWidget(content)
        return card

    def _wrap_battery(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        cap = QLabel("BATTERY")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        h = QHBoxLayout()
        h.addWidget(self._battery_pct)
        h.addStretch(1)
        h.addWidget(self._battery_volts)
        v.addLayout(h)
        v.addWidget(self._plot_batt, 1)
        return card

    def _wrap_bumpcliff(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        cap = QLabel("SAFETY SENSORS")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        row = QHBoxLayout()
        row.setSpacing(20)
        row.addLayout(self._led_row("Bumpers", self._bump_leds))
        row.addLayout(self._led_row("Cliff",   self._cliff_leds))
        row.addStretch(1)
        v.addLayout(row)
        return card

    def _led_row(self, name: str, leds: list) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(8)
        title = QLabel(name)
        title.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-weight: 600;")
        h.addWidget(title)
        for label, led in zip(("L", "C", "R"), leds):
            wrap = QVBoxLayout()
            wrap.setSpacing(2)
            wrap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            wrap.addWidget(led, alignment=Qt.AlignmentFlag.AlignHCenter)
            l = QLabel(label)
            l.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 10px;")
            wrap.addWidget(l, alignment=Qt.AlignmentFlag.AlignHCenter)
            h.addLayout(wrap)
        return h

    # ---------------------------------------------------------------

    def _wire_signals(self) -> None:
        if self.ros is None:
            return
        self.ros.imu_updated.connect(self._on_imu)
        self.ros.encoders_updated.connect(self._on_encoders)
        self.ros.battery_updated.connect(self._on_battery)
        self.ros.bump_event.connect(self._on_bumper)
        self.ros.cliff_event.connect(self._on_cliff)

    # ---- ROS signal handlers (run on the GUI thread thanks to queued conn) ----

    def _on_imu(self) -> None:
        with self.state.lock:
            imu = self.state.imu
        a = imu.linear_acceleration
        g = imu.angular_velocity
        for buf, val in zip(self._buf_accel, a):
            buf.append(float(val))
        for buf, val in zip(self._buf_gyro, g):
            buf.append(float(val))

    def _on_encoders(self) -> None:
        import time
        with self.state.lock:
            l, r = self.state.encoders_lr
        now = time.monotonic()
        if self._last_enc_l is not None and self._last_enc_t > 0:
            dt = max(1e-3, now - self._last_enc_t)
            v_l = (l - self._last_enc_l) / 2578.0 / dt
            v_r = (r - self._last_enc_r) / 2578.0 / dt
            self._buf_vel_l.append(v_l)
            self._buf_vel_r.append(v_r)
        self._last_enc_l, self._last_enc_r, self._last_enc_t = l, r, now

    def _on_battery(self) -> None:
        with self.state.lock:
            v = self.state.battery.voltage
            p = self.state.battery.percent
        self._buf_battery_v.append(v)
        self._battery_pct.setText(f"{int(p)}%")
        self._battery_volts.setText(f"{v:.2f} V")
        # Tone the percentage colour
        if p > 60:
            self._battery_pct.setStyleSheet(f"color: {Tokens.SUCCESS};")
        elif p > 25:
            self._battery_pct.setStyleSheet(f"color: {Tokens.WARNING};")
        else:
            self._battery_pct.setStyleSheet(f"color: {Tokens.DANGER};")

    def _on_bumper(self, idx: int, active: bool) -> None:
        if idx < 0:                  # all-clear event
            for led in self._bump_leds:
                led.set_active(False)
            return
        if 0 <= idx < len(self._bump_leds):
            self._bump_leds[idx].set_active(active, tone="bad")

    def _on_cliff(self, idx: int, active: bool) -> None:
        if idx < 0:
            for led in self._cliff_leds:
                led.set_active(False)
            return
        if 0 <= idx < len(self._cliff_leds):
            self._cliff_leds[idx].set_active(active, tone="warn")

    # ---- Periodic plot redraw ----

    def _redraw(self) -> None:
        # Accel
        n = len(self._buf_accel[0])
        if n >= 2:
            xs = list(range(n))
            self._curve_ax.setData(xs, list(self._buf_accel[0]))
            self._curve_ay.setData(xs, list(self._buf_accel[1]))
            self._curve_az.setData(xs, list(self._buf_accel[2]))
        n = len(self._buf_gyro[0])
        if n >= 2:
            xs = list(range(n))
            self._curve_gx.setData(xs, list(self._buf_gyro[0]))
            self._curve_gy.setData(xs, list(self._buf_gyro[1]))
            self._curve_gz.setData(xs, list(self._buf_gyro[2]))
        if len(self._buf_vel_l) >= 2:
            xs = list(range(len(self._buf_vel_l)))
            self._curve_vl.setData(xs, list(self._buf_vel_l))
            self._curve_vr.setData(xs, list(self._buf_vel_r))
        if len(self._buf_battery_v) >= 2:
            xs = list(range(len(self._buf_battery_v)))
            self._curve_bv.setData(xs, list(self._buf_battery_v))
