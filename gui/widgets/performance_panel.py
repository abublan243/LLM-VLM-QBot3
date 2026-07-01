"""
PerformancePanelWidget — real-time pipeline and system metrics dashboard.

Subscribes to ``PerformanceMonitor.metrics_updated(dict)`` and renders:
  * 4 pyqtgraph line plots: latency trio, FPS, CPU/mem, WebSocket RTT
  * Big-number readouts: tokens, success rate, battery
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Tokens

# ---------- pyqtgraph global theming (idempotent) ----------
pg.setConfigOptions(antialias=True, background=None, foreground=Tokens.TEXT_SECONDARY)

_HISTORY = 120          # samples (≈ 60 s @ 500 ms)
_PEN_WIDTH = 2


def _accent_pen(color: str) -> pg.mkPen:  # type: ignore[override]
    return pg.mkPen(color=color, width=_PEN_WIDTH)


class PerformancePanelWidget(QWidget):
    """Mode-agnostic performance dashboard."""

    def __init__(self, *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Sliding-window stores
        self._vlm_lat: Deque[float] = deque(maxlen=_HISTORY)
        self._llm_lat: Deque[float] = deque(maxlen=_HISTORY)
        self._mot_lat: Deque[float] = deque(maxlen=_HISTORY)
        self._yolo_fps: Deque[float] = deque(maxlen=_HISTORY)
        self._cam_fps: Deque[float] = deque(maxlen=_HISTORY)
        self._cpu: Deque[float] = deque(maxlen=_HISTORY)
        self._mem: Deque[float] = deque(maxlen=_HISTORY)
        self._rtt: Deque[float] = deque(maxlen=_HISTORY)

        self._build()

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # 2×2 plot grid
        grid = QGridLayout()
        grid.setSpacing(8)

        self._plot_latency, self._curves_latency = self._make_plot(
            "Pipeline Latency (ms)",
            [("VLM", Tokens.ACCENT_PRIMARY),
             ("LLM", Tokens.ACCENT_SECONDARY),
             ("Motor", Tokens.WARNING)],
        )
        grid.addWidget(self._plot_latency, 0, 0)

        self._plot_fps, self._curves_fps = self._make_plot(
            "Frame Rate (FPS)",
            [("YOLO", Tokens.ACCENT_PRIMARY),
             ("Camera", Tokens.ACCENT_SECONDARY)],
        )
        grid.addWidget(self._plot_fps, 0, 1)

        self._plot_sys, self._curves_sys = self._make_plot(
            "System (%)",
            [("CPU", Tokens.DANGER),
             ("Memory", Tokens.INFO)],
        )
        grid.addWidget(self._plot_sys, 1, 0)

        self._plot_rtt, self._curves_rtt = self._make_plot(
            "DDS Topic Age (ms)",
            [("IMU age", Tokens.ACCENT_SECONDARY)],
        )
        grid.addWidget(self._plot_rtt, 1, 1)

        outer.addLayout(grid, 1)

        # Big-number readout row
        readout_card = QFrame()
        readout_card.setProperty("role", "card")
        rh = QHBoxLayout(readout_card)
        rh.setContentsMargins(16, 10, 16, 10)
        rh.setSpacing(24)

        self._tokens_label = self._big_number("0", "Tokens")
        rh.addWidget(self._tokens_label)
        self._success_label = self._big_number("—", "Success Rate")
        rh.addWidget(self._success_label)
        self._battery_label = self._big_number("—", "Battery")
        rh.addWidget(self._battery_label)

        outer.addWidget(readout_card)

        # ---- Numeric stats table — current / min / avg / max for every metric ----
        # Operators want exact numbers, not just graph eyeballing. This table
        # is updated each time set_metrics() is called. Min/avg/max are
        # computed from the same sliding deques the graphs use.
        outer.addWidget(self._build_stats_card())

    # ---------------------------------------------------------------
    # Stats card factory
    # ---------------------------------------------------------------

    # Each row binds a deque-attribute name to (label, unit, value formatter).
    _STAT_ROWS = (
        ("_vlm_lat",  "VLM latency",     "ms",  "{:6.1f}"),
        ("_llm_lat",  "LLM latency",     "ms",  "{:6.1f}"),
        ("_mot_lat",  "Motor cmd",       "ms",  "{:6.1f}"),
        ("_yolo_fps", "YOLO",            "fps", "{:5.1f}"),
        ("_cam_fps",  "Camera",          "fps", "{:5.1f}"),
        ("_cpu",      "CPU",             "%",   "{:5.1f}"),
        ("_mem",      "Memory",          "%",   "{:5.1f}"),
        ("_rtt",      "DDS topic age",   "ms",  "{:6.1f}"),
    )

    def _build_stats_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 10, 16, 12)
        v.setSpacing(6)

        cap = QLabel("METRICS — current / min / avg / max")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)

        # Header row
        for col, (text, align) in enumerate((
            ("Metric", Qt.AlignmentFlag.AlignLeft),
            ("Current", Qt.AlignmentFlag.AlignRight),
            ("Min", Qt.AlignmentFlag.AlignRight),
            ("Avg", Qt.AlignmentFlag.AlignRight),
            ("Max", Qt.AlignmentFlag.AlignRight),
            ("Unit", Qt.AlignmentFlag.AlignLeft),
        )):
            h = QLabel(text)
            h.setStyleSheet(
                f"color: {Tokens.TEXT_MUTED}; font-size: 10px; "
                "letter-spacing: 0.6px; text-transform: uppercase;"
            )
            h.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(h, 0, col)

        # Pre-allocate value labels for every row × column
        self._stat_labels: dict = {}
        for row_idx, (attr, label, unit, _fmt) in enumerate(self._STAT_ROWS, start=1):
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(
                f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;"
            )
            grid.addWidget(name_lbl, row_idx, 0)

            cells = []
            for col_idx in range(1, 5):       # current, min, avg, max
                cell = QLabel("—")
                cell.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                cell.setStyleSheet(
                    f"color: {Tokens.TEXT_PRIMARY}; "
                    f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
                )
                grid.addWidget(cell, row_idx, col_idx)
                cells.append(cell)

            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 10px;")
            grid.addWidget(unit_lbl, row_idx, 5)

            self._stat_labels[attr] = cells

        # Make the value columns share width evenly
        for col in range(1, 5):
            grid.setColumnStretch(col, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(5, 1)

        v.addLayout(grid)
        return card

    @staticmethod
    def _stats_for(deque_obj) -> tuple:
        """Return (current, min, avg, max) for a sliding-window deque.
        Skips zeros for latency-style deques where 0 means 'not yet measured'.
        """
        if not deque_obj:
            return None
        values = [v for v in deque_obj if v is not None]
        if not values:
            return None
        # Ignore leading/initial zeros that bias the min — but keep them in
        # avg so the user sees a realistic warm-up reading.
        nonzero = [v for v in values if v != 0.0]
        cur = values[-1]
        lo = min(nonzero) if nonzero else min(values)
        hi = max(values)
        avg = sum(values) / len(values)
        return cur, lo, avg, hi

    def _update_stats_table(self) -> None:
        for attr, _label, _unit, fmt in self._STAT_ROWS:
            stats = self._stats_for(getattr(self, attr, None))
            cells = self._stat_labels.get(attr)
            if cells is None:
                continue
            if stats is None:
                for cell in cells:
                    cell.setText("—")
                continue
            cur, lo, avg, hi = stats
            for cell, value in zip(cells, (cur, lo, avg, hi)):
                cell.setText(fmt.format(value))

    # ---------------------------------------------------------------
    # Plot factory
    # ---------------------------------------------------------------

    @staticmethod
    def _make_plot(title: str, curves: list) -> tuple:
        """Return (PlotWidget, {name: PlotDataItem}) for a themed pyqtgraph plot."""
        pw = pg.PlotWidget()
        pw.setBackground(Tokens.SURFACE)
        pw.setTitle(title, color=Tokens.TEXT_SECONDARY, size="10pt")
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.setMouseEnabled(x=False, y=False)
        pw.hideButtons()
        pw.getPlotItem().getViewBox().setDefaultPadding(0.02)

        # Axis styling
        for axis_name in ("left", "bottom"):
            ax = pw.getPlotItem().getAxis(axis_name)
            ax.setPen(pg.mkPen(color=Tokens.BORDER_BRIGHT, width=1))
            ax.setTextPen(pg.mkPen(color=Tokens.TEXT_MUTED))

        curve_map: Dict[str, pg.PlotDataItem] = {}
        for name, color in curves:
            item = pw.plot(pen=_accent_pen(color), name=name)
            curve_map[name] = item

        return pw, curve_map

    # ---------------------------------------------------------------
    # Big-number helper
    # ---------------------------------------------------------------

    @staticmethod
    def _big_number(value: str, caption: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        num = QLabel(value)
        num.setObjectName(caption.replace(" ", "_").lower() + "_value")
        num.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY};"
            f"font-family: {Tokens.FONT_FAMILY_MONO};"
            "font-size: 22px; font-weight: 700;"
        )
        num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(num)

        cap = QLabel(caption)
        cap.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-size: 10px;"
            "letter-spacing: 0.5px; text-transform: uppercase;"
        )
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(cap)
        return w

    # ---------------------------------------------------------------
    # Public slot — wire to PerformanceMonitor.metrics_updated
    # ---------------------------------------------------------------

    def set_metrics(self, m: Dict[str, float]) -> None:
        """Push one metrics snapshot and redraw all curves."""
        self._vlm_lat.append(m.get("vlm_latency_ms", 0.0))
        self._llm_lat.append(m.get("llm_latency_ms", 0.0))
        self._mot_lat.append(m.get("motor_cmd_latency_ms", 0.0))
        self._yolo_fps.append(m.get("yolo_fps", 0.0))
        self._cam_fps.append(m.get("camera_fps", 0.0))
        self._cpu.append(m.get("cpu_percent", 0.0))
        self._mem.append(m.get("memory_percent", 0.0))
        self._rtt.append(m.get("websocket_rtt_ms", 0.0))

        # Update curves
        self._curves_latency["VLM"].setData(list(self._vlm_lat))
        self._curves_latency["LLM"].setData(list(self._llm_lat))
        self._curves_latency["Motor"].setData(list(self._mot_lat))

        self._curves_fps["YOLO"].setData(list(self._yolo_fps))
        self._curves_fps["Camera"].setData(list(self._cam_fps))

        self._curves_sys["CPU"].setData(list(self._cpu))
        self._curves_sys["Memory"].setData(list(self._mem))

        self._curves_rtt["IMU age"].setData(list(self._rtt))

        # Numeric stats table — refresh every push
        self._update_stats_table()

        # Big numbers
        tokens = int(m.get("tokens_used_session", 0))
        self._tokens_label.findChild(QLabel, "tokens_value").setText(
            f"{tokens:,}" if tokens < 1_000_000 else f"{tokens/1e6:.1f}M"
        )

        attempted = int(m.get("task_attempted", 0))
        rate = m.get("task_success_rate", 0.0)
        self._success_label.findChild(QLabel, "success_rate_value").setText(
            f"{rate:.0%}" if attempted > 0 else "—"
        )

        batt = m.get("battery_percent", 0.0)
        batt_lbl = self._battery_label.findChild(QLabel, "battery_value")
        batt_lbl.setText(f"{batt:.0f}%")
        if batt > 50:
            batt_lbl.setStyleSheet(
                f"color: {Tokens.SUCCESS}; font-family: {Tokens.FONT_FAMILY_MONO};"
                "font-size: 22px; font-weight: 700;"
            )
        elif batt > 20:
            batt_lbl.setStyleSheet(
                f"color: {Tokens.WARNING}; font-family: {Tokens.FONT_FAMILY_MONO};"
                "font-size: 22px; font-weight: 700;"
            )
        else:
            batt_lbl.setStyleSheet(
                f"color: {Tokens.DANGER}; font-family: {Tokens.FONT_FAMILY_MONO};"
                "font-size: 22px; font-weight: 700;"
            )
