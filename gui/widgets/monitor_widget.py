"""
MonitorWidget — operator-configurable 2×2 monitoring grid.

Each cell has a dropdown so the operator picks what they want to see in
that quadrant. Choices:

    RGB + YOLO    — live colour frame with YOLO overlays
    Depth         — colorised depth heatmap
    SLAM map      — host-side occupancy grid
    Sensors       — compact battery + bump/cliff + IMU summary
    AI Thought    — last VLM scene description + LLM reasoning excerpt
    Performance   — numeric snapshot of pipeline + system metrics

Each mini widget is light, self-contained, pulls straight from SharedState
or PerformanceMonitor signals, and pauses its refresh timer when hidden.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens
from gui.widgets.camera_viewer import AspectImageLabel

logger = logging.getLogger(__name__)


# =====================================================================
# Mini widgets — one per content type
# =====================================================================


class _MiniBase(QWidget):
    """Base for mini widgets: refresh timer that pauses on hide."""

    REFRESH_MS = 250

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._refresh)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
            self._refresh()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _refresh(self) -> None:        # subclasses override
        pass


# ---- RGB + YOLO mini ----


class _MiniRGB(_MiniBase):
    REFRESH_MS = 80     # ~12 FPS preview is plenty for a small monitor cell

    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._image = AspectImageLabel(self, placeholder="WAITING FOR CAMERA")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._image)

    def _refresh(self) -> None:
        with self.state.lock:
            frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            detections = list(self.state.detected_objects)
        if frame is None:
            self._image.clear_image()
            return
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 99, 108), 2)
            label = f"{det.class_name} {int(det.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1),
                          (255, 99, 108), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 245), 1, cv2.LINE_AA)
        self._image.set_bgr(frame)


# ---- Depth mini ----


class _MiniDepth(_MiniBase):
    REFRESH_MS = 120

    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._image = AspectImageLabel(self, placeholder="WAITING FOR DEPTH")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._image)

    def _refresh(self) -> None:
        with self.state.lock:
            visual = (
                None if self.state.depth_visual_frame is None
                else self.state.depth_visual_frame.copy()
            )
            depth = (
                None if self.state.depth_frame is None
                else self.state.depth_frame.copy()
            )
        if visual is None and depth is not None:
            clipped = np.clip(depth, 0, 3000).astype(np.float32) / 3000.0
            inv = (255 - clipped * 255).astype(np.uint8)
            visual = cv2.applyColorMap(inv, cv2.COLORMAP_JET)
            visual[depth == 0] = (0, 0, 0)
        if visual is None:
            self._image.clear_image()
            return
        self._image.set_bgr(visual)


# ---- SLAM mini ----


class _MiniSLAM(_MiniBase):
    REFRESH_MS = 300

    def __init__(self, slam_manager: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._slam = slam_manager
        self._image = AspectImageLabel(self, placeholder="MAPPING…")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._image)

    def _refresh(self) -> None:
        if self._slam is None:
            return
        try:
            img = self._slam.get_map_image()
        except Exception:
            return
        if img is not None and img.size > 0:
            self._image.set_bgr(img)


# ---- Sensors mini (compact summary) ----


class _MiniSensors(_MiniBase):
    REFRESH_MS = 250

    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        # Battery
        self._battery_pct = QLabel("--%")
        self._battery_pct.setProperty("role", "display")
        self._battery_volts = QLabel("0.00 V")
        self._battery_volts.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        batt_row = QHBoxLayout()
        batt_lbl = QLabel("BATTERY")
        batt_lbl.setProperty("role", "caption")
        batt_row.addWidget(batt_lbl)
        batt_row.addStretch(1)
        batt_row.addWidget(self._battery_pct)
        batt_row.addWidget(self._battery_volts)
        v.addLayout(batt_row)

        # Pose
        self._pose = QLabel("x=0.00  y=0.00  yaw=0°")
        self._pose.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 12px;"
        )
        v.addWidget(self._pose)

        # IMU summary (a/g magnitude)
        self._imu = QLabel("|a|=9.80 m/s²   |ω|=0.00 rad/s")
        self._imu.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        v.addWidget(self._imu)

        # Bumpers + cliff dots
        self._bump = self._dot_row("Bumpers", ("L", "C", "R"))
        self._cliff = self._dot_row("Cliff", ("L", "C", "R"))
        v.addLayout(self._bump["row"])
        v.addLayout(self._cliff["row"])

        # Connection
        self._conn = QLabel("disconnected")
        self._conn.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-size: 11px;"
        )
        v.addWidget(self._conn)

        v.addStretch(1)

    def _dot_row(self, label: str, names: tuple) -> Dict[str, Any]:
        h = QHBoxLayout()
        h.setSpacing(6)
        ll = QLabel(label)
        ll.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-weight: 600; font-size: 11px;")
        ll.setMinimumWidth(64)
        h.addWidget(ll)
        dots: List[QLabel] = []
        for _ in names:
            dot = QLabel()
            dot.setFixedSize(14, 14)
            self._set_dot(dot, False)
            dots.append(dot)
            h.addWidget(dot)
        h.addStretch(1)
        return {"row": h, "dots": dots}

    @staticmethod
    def _set_dot(dot: QLabel, active: bool, tone: str = "bad") -> None:
        if active:
            color = {"bad": Tokens.DANGER, "warn": Tokens.WARNING}.get(tone, Tokens.DANGER)
            dot.setStyleSheet(
                f"background: {color}; border-radius: 7px;"
                f"border: 2px solid rgba(255,255,255,30);"
            )
        else:
            dot.setStyleSheet(
                f"background: {Tokens.SURFACE_ELEVATED}; border-radius: 7px;"
                f"border: 1px solid {Tokens.BORDER};"
            )

    def _refresh(self) -> None:
        with self.state.lock:
            batt = self.state.battery
            o = self.state.odom
            imu = self.state.imu
            bumps = self.state.bumpers.active()
            cliffs = self.state.cliff.active()
            connected = self.state.ros_connected
            conn_msg = self.state.ros_status_message

        self._battery_pct.setText(f"{int(batt.percent)}%")
        self._battery_volts.setText(f"{batt.voltage:.2f} V")
        if batt.percent > 60:
            self._battery_pct.setStyleSheet(
                f"color: {Tokens.SUCCESS}; font-size: 22px; font-weight: 700; "
                f"font-family: {Tokens.FONT_FAMILY_MONO};"
            )
        elif batt.percent > 25:
            self._battery_pct.setStyleSheet(
                f"color: {Tokens.WARNING}; font-size: 22px; font-weight: 700; "
                f"font-family: {Tokens.FONT_FAMILY_MONO};"
            )
        else:
            self._battery_pct.setStyleSheet(
                f"color: {Tokens.DANGER}; font-size: 22px; font-weight: 700; "
                f"font-family: {Tokens.FONT_FAMILY_MONO};"
            )

        self._pose.setText(
            f"x={o.x:+.2f}  y={o.y:+.2f}  yaw={math.degrees(o.yaw_rad):+.0f}°"
        )
        a_mag = math.sqrt(sum(v * v for v in imu.linear_acceleration))
        w_mag = math.sqrt(sum(v * v for v in imu.angular_velocity))
        self._imu.setText(f"|a|={a_mag:5.2f} m/s²   |ω|={w_mag:5.2f} rad/s")

        for dot, active in zip(self._bump["dots"], bumps):
            self._set_dot(dot, active, tone="bad")
        for dot, active in zip(self._cliff["dots"], cliffs):
            self._set_dot(dot, active, tone="warn")

        if connected:
            self._conn.setStyleSheet(f"color: {Tokens.SUCCESS}; font-size: 11px;")
            self._conn.setText("● connected · " + conn_msg)
        else:
            self._conn.setStyleSheet(f"color: {Tokens.DANGER}; font-size: 11px;")
            self._conn.setText("○ " + conn_msg)


# ---- AI Thought mini ----


class _MiniThought(_MiniBase):
    REFRESH_MS = 500

    def __init__(self, state: SharedState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        cap = QLabel("AI THOUGHT")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        self._meta = QLabel("waiting for first plan…")
        self._meta.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 10px;"
        )
        v.addWidget(self._meta)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setProperty("role", "log")
        self._text.setPlaceholderText("(no plan yet)")
        v.addWidget(self._text, 1)

    def _refresh(self) -> None:
        with self.state.lock:
            llm = self.state.llm_last_output
            vlm = self.state.vlm_last_output

        lines: List[str] = []
        if llm is not None:
            lines.append(f"STATUS: {llm.status or '—'}   conf={llm.confidence:.2f}")
            if llm.reasoning:
                lines.append("REASONING:")
                lines.append(llm.reasoning[:600])
            if llm.action_type == "low_level" and llm.low_level_command:
                cmd = llm.low_level_command
                lines.append("")
                lines.append(
                    f"→ low_level: lin={cmd.get('linear_x', 0.0):+.2f} "
                    f"ang={cmd.get('angular_z', 0.0):+.2f} "
                    f"dt={cmd.get('duration_ms', 0)} ms"
                )
            elif llm.action_type == "skill" and llm.skill_command:
                lines.append("")
                lines.append(f"→ skill: {llm.skill_command.get('skill_name', '?')}")
            self._meta.setText(
                f"LLM {llm.model}  ·  {llm.latency_ms:.0f} ms"
                + (f"  ·  VLM {vlm.model} {vlm.latency_ms:.0f} ms" if vlm else "")
            )
        elif vlm is not None:
            lines.append(f"VLM scene ({vlm.model}, {vlm.latency_ms:.0f} ms):")
            lines.append(vlm.scene_description or vlm.raw_text[:600])
            self._meta.setText(f"VLM only — no LLM plan yet")
        text = "\n".join(lines)
        if self._text.toPlainText() != text:
            sb = self._text.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 8
            self._text.setPlainText(text)
            if at_bottom:
                sb.setValue(sb.maximum())


# ---- Performance mini (numeric snapshot, no graphs) ----


class _MiniPerf(_MiniBase):
    REFRESH_MS = 600

    def __init__(self, perf_monitor: Optional[Any] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._perf = perf_monitor
        self._latest: Dict[str, float] = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(8)

        cap = QLabel("PERFORMANCE")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)

        self._labels: Dict[str, QLabel] = {}
        rows = [
            ("vlm_latency_ms",   "VLM latency",     "ms"),
            ("llm_latency_ms",   "LLM latency",     "ms"),
            ("motor_cmd_latency_ms", "Motor cmd",   "ms"),
            ("yolo_fps",         "YOLO",            "fps"),
            ("camera_fps",       "Camera",          "fps"),
            ("cpu_percent",      "CPU",             "%"),
            ("memory_percent",   "Memory",          "%"),
            ("websocket_rtt_ms", "DDS topic age",   "ms"),
            ("battery_percent",  "Battery",         "%"),
            ("task_success_rate", "Task success",   ""),
            ("tokens_used_session", "Tokens",       ""),
        ]
        for i, (key, label, unit) in enumerate(rows):
            l_lbl = QLabel(label)
            l_lbl.setStyleSheet(
                f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;"
            )
            v_lbl = QLabel("—")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v_lbl.setStyleSheet(
                f"color: {Tokens.TEXT_PRIMARY}; "
                f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 12px; font-weight: 600;"
            )
            u_lbl = QLabel(unit)
            u_lbl.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 10px;")
            grid.addWidget(l_lbl, i, 0)
            grid.addWidget(v_lbl, i, 1)
            grid.addWidget(u_lbl, i, 2)
            self._labels[key] = v_lbl

        v.addLayout(grid)
        v.addStretch(1)

        if perf_monitor is not None:
            try:
                perf_monitor.metrics_updated.connect(self._on_metrics)
            except Exception:
                pass

    def _on_metrics(self, m: Dict[str, float]) -> None:
        self._latest = dict(m)

    def _refresh(self) -> None:
        m = self._latest
        if not m:
            return
        for key, lbl in self._labels.items():
            v = m.get(key, 0.0)
            if key == "task_success_rate":
                txt = f"{v * 100:.0f}%" if m.get("task_attempted", 0) else "—"
            elif key == "tokens_used_session":
                t = int(v)
                txt = f"{t:,}" if t < 1_000_000 else f"{t/1e6:.1f}M"
            elif key in ("yolo_fps", "camera_fps"):
                txt = f"{v:.1f}"
            elif key in ("cpu_percent", "memory_percent", "battery_percent"):
                txt = f"{v:.0f}"
            else:
                txt = f"{v:.0f}"
            lbl.setText(txt)


# =====================================================================
# MonitorCell — header dropdown + stacked content
# =====================================================================


_CONTENT_OPTIONS = (
    ("RGB + YOLO",    "rgb"),
    ("Depth",         "depth"),
    ("SLAM map",      "slam"),
    ("Sensors",       "sensors"),
    ("AI Thought",    "thought"),
    ("Performance",   "perf"),
)


class _MonitorCell(QWidget):
    """One quadrant: dropdown header + stacked content widgets."""

    def __init__(
        self,
        state: SharedState,
        slam_manager: Any,
        perf_monitor: Optional[Any],
        *,
        default_kind: str = "rgb",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.slam_manager = slam_manager
        self.perf_monitor = perf_monitor

        outer = QFrame()
        outer.setProperty("role", "card")
        ov = QVBoxLayout(outer)
        ov.setContentsMargins(8, 6, 8, 8)
        ov.setSpacing(4)

        # Header: small caption + dropdown
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        cap = QLabel("Show")
        cap.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-size: 10px; "
            "letter-spacing: 0.6px; text-transform: uppercase;"
        )
        hdr.addWidget(cap)
        self._combo = QComboBox()
        for label, key in _CONTENT_OPTIONS:
            self._combo.addItem(label, key)
        idx = next((i for i, (_, k) in enumerate(_CONTENT_OPTIONS)
                    if k == default_kind), 0)
        self._combo.setCurrentIndex(idx)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        hdr.addWidget(self._combo, 1)
        ov.addLayout(hdr)

        # Stacked content; mini widgets created lazily on first selection
        self._stack = QStackedWidget()
        self._stack_idx: Dict[str, int] = {}
        ov.addWidget(self._stack, 1)

        self._activate(default_kind)

        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(outer)

    def _on_combo_changed(self, _idx: int) -> None:
        kind = self._combo.currentData()
        self._activate(kind)

    def _activate(self, kind: str) -> None:
        if kind in self._stack_idx:
            self._stack.setCurrentIndex(self._stack_idx[kind])
            return
        widget = self._make(kind)
        idx = self._stack.addWidget(widget)
        self._stack_idx[kind] = idx
        self._stack.setCurrentIndex(idx)

    def _make(self, kind: str) -> QWidget:
        if kind == "rgb":
            return _MiniRGB(self.state)
        if kind == "depth":
            return _MiniDepth(self.state)
        if kind == "slam":
            return _MiniSLAM(self.slam_manager)
        if kind == "sensors":
            return _MiniSensors(self.state)
        if kind == "thought":
            return _MiniThought(self.state)
        if kind == "perf":
            return _MiniPerf(self.perf_monitor)
        # Fallback
        lbl = QLabel("(unknown)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl


# =====================================================================
# MonitorWidget — top-level 2×2 grid
# =====================================================================


class MonitorWidget(QWidget):
    """Top-level Monitor tab: configurable 2×2 grid of mini widgets."""

    def __init__(
        self,
        state: SharedState,
        slam_manager: Any,
        perf_monitor: Optional[Any] = None,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Sensible default layout: vision in front, sensors on the side,
        # AI thought + performance below — all of the most-watched signals
        # in one screen.
        defaults = ("rgb", "slam", "sensors", "thought")

        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        self._cells: List[_MonitorCell] = []
        for i, default in enumerate(defaults):
            cell = _MonitorCell(state, slam_manager, perf_monitor,
                                default_kind=default, parent=self)
            self._cells.append(cell)
            row, col = divmod(i, 2)
            grid.addWidget(cell, row, col)

        for r in range(2):
            grid.setRowStretch(r, 1)
        for c in range(2):
            grid.setColumnStretch(c, 1)
