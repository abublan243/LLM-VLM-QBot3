"""
VisionWidget — dedicated full-screen RGB + YOLO view.

Left  (~70 %): live colour frame with YOLO bounding boxes, class labels,
               confidence, and depth-derived distance overlaid.
Right (~30 %): scrollable detections list (class | conf | distance), top-3
               highlighted, plus a counter and a target-class quick-filter
               that hides everything except a chosen class.

This sits as a top-level tab in the main window so the operator can see the
camera + detector at full size without going through the tabbed Camera card.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens
from gui.widgets.camera_viewer import AspectImageLabel

logger = logging.getLogger(__name__)


class VisionWidget(QWidget):
    """Top-level RGB + YOLO viewer with a side-panel detection list."""

    # External hook the MainWindow installs so toggles survive restart.
    # Callable signature: (layer_key: str, enabled: bool) -> None
    persist_layer_change: Optional[Any] = None

    def __init__(
        self,
        state: SharedState,
        *,
        vlm_pipeline: Optional[Any] = None,
        display_fps: int = 20,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._vlm = vlm_pipeline
        self._filter_class: str = ""
        self._known_classes: set = set()

        self._build()

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / max(1, display_fps)))
        self._timer.timeout.connect(self._refresh)

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)

        # ---- Left: live RGB + YOLO overlay ----
        viewer_card = QFrame()
        viewer_card.setProperty("role", "card")
        vv = QVBoxLayout(viewer_card)
        vv.setContentsMargins(12, 12, 12, 12)
        vv.setSpacing(6)

        title = QLabel("LIVE RGB + YOLO")
        title.setProperty("role", "caption")
        vv.addWidget(title)

        # Detection-layer toggles — runtime enable/disable of each layer.
        # Each disabled layer pays zero inference cost on the next frame,
        # so this is the operator's primary lever to recover FPS.
        vv.addLayout(self._build_layer_toggles())

        self._image = AspectImageLabel(self, placeholder="WAITING FOR CAMERA")
        vv.addWidget(self._image, 1)

        # Inline status strip below the image
        status_row = QHBoxLayout()
        status_row.setSpacing(12)
        self._fps_label = QLabel("0 FPS  ·  0 detections")
        self._fps_label.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        status_row.addWidget(self._fps_label)
        status_row.addStretch(1)

        # Class quick-filter so the operator can isolate one class
        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All classes")
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._filter_combo.setMinimumWidth(160)
        status_row.addWidget(QLabel("Filter:"))
        status_row.addWidget(self._filter_combo)
        vv.addLayout(status_row)

        split.addWidget(viewer_card)

        # ---- Right: detection list ----
        det_card = QFrame()
        det_card.setProperty("role", "card")
        dv = QVBoxLayout(det_card)
        dv.setContentsMargins(12, 12, 12, 12)
        dv.setSpacing(8)

        det_cap = QLabel("DETECTIONS")
        det_cap.setProperty("role", "caption")
        dv.addWidget(det_cap)

        self._summary = QLabel("0 objects in view")
        self._summary.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 13px; font-weight: 600;"
        )
        dv.addWidget(self._summary)

        self._list = QListWidget()
        dv.addWidget(self._list, 1)
        det_card.setMinimumWidth(240)
        split.addWidget(det_card)

        split.setStretchFactor(0, 7)
        split.setStretchFactor(1, 3)

    # ---------------------------------------------------------------
    # Detection-layer toggles
    # ---------------------------------------------------------------

    def _build_layer_toggles(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(14)

        cap = QLabel("Detection layers:")
        cap.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; font-family: {Tokens.FONT_FAMILY_MONO}; "
            f"font-size: 11px;"
        )
        row.addWidget(cap)

        states = (
            self._vlm.get_layer_states()
            if self._vlm is not None
            else {"coco": True, "yolo_world": True}
        )

        self._cb_coco = self._make_layer_checkbox(
            "● COCO (yolo11)", "coco",
            "Primary YOLO model (yolo11n/l/x). ~5-40ms per frame.",
            states.get("coco", True),
        )
        row.addWidget(self._cb_coco)

        self._cb_yw = self._make_layer_checkbox(
            "● YOLO-World (yw_*)", "yolo_world",
            "Open-vocabulary detector. ~15-25ms per frame.",
            states.get("yolo_world", True),
        )
        row.addWidget(self._cb_yw)

        row.addStretch(1)
        return row

    def _make_layer_checkbox(self, label: str, key: str, tip: str,
                             checked: bool) -> "QCheckBox":
        cb = QCheckBox(label)
        cb.setChecked(bool(checked))
        cb.setToolTip(tip)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {Tokens.TEXT_PRIMARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px; "
            f"spacing: 6px; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        cb.toggled.connect(lambda on, k=key: self._on_layer_toggled(k, on))
        return cb

    def _on_layer_toggled(self, key: str, enabled: bool) -> None:
        if self._vlm is not None:
            try:
                self._vlm.set_layer_enabled(key, enabled)
            except Exception as exc:
                logger.exception("set_layer_enabled(%s, %s) failed: %s",
                                 key, enabled, exc)
                return
        # Persist across restarts via the MainWindow-installed callback
        cb = self.persist_layer_change
        if callable(cb):
            try:
                cb(key, enabled)
            except Exception:
                logger.exception("persist_layer_change failed for %s", key)

    # ---------------------------------------------------------------
    # Show / hide gating
    # ---------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
            self._refresh()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    # ---------------------------------------------------------------
    # Refresh tick
    # ---------------------------------------------------------------

    def _refresh(self) -> None:
        with self.state.lock:
            frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            detections = list(self.state.detected_objects)

        if frame is None:
            self._image.clear_image()
            self._summary.setText("waiting for camera…")
            self._list.clear()
            return

        # Filter by selected class if the user picked one
        filtered = (
            [d for d in detections if d.class_name.lower() == self._filter_class]
            if self._filter_class else detections
        )

        # ---- Draw overlays ----
        self._draw_overlays(frame, filtered)
        self._image.set_bgr(frame)

        # ---- Update side panel ----
        n = len(filtered)
        total = len(detections)
        if self._filter_class:
            self._summary.setText(
                f"{n} '{self._filter_class}' shown   ·   {total} total"
            )
        else:
            self._summary.setText(f"{total} object{'s' if total != 1 else ''} in view")
        self._fps_label.setText(
            f"{1000 // max(1, self._timer.interval())} FPS  ·  {total} detections"
        )

        self._list.clear()
        # Sort detections by confidence descending so the strongest hits are on top
        for det in sorted(filtered, key=lambda d: -d.confidence):
            dist_str = f"{det.distance_m:.2f}m" if det.distance_m and det.distance_m > 0 else "—"
            text = f"{det.class_name:14} {int(det.confidence*100):3d}%  {dist_str}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(Tokens.TEXT_PRIMARY))
            # Tone the item background by confidence (stronger = more accent)
            tint = QColor(Tokens.ACCENT_PRIMARY)
            tint.setAlpha(int(40 + 80 * min(1.0, det.confidence)))
            item.setBackground(tint)
            item.setToolTip(
                f"{det.class_name}\n"
                f"confidence: {det.confidence:.3f}\n"
                f"distance: {dist_str}\n"
                f"bbox: {det.bbox_xyxy}\n"
                f"3D: {det.position_3d}"
            )
            self._list.addItem(item)

        # Maintain the filter combo's class list
        new_classes = {d.class_name for d in detections}
        if new_classes != self._known_classes:
            self._known_classes = new_classes
            self._rebuild_filter_combo()

    def _rebuild_filter_combo(self) -> None:
        prev = self._filter_class
        self._filter_combo.blockSignals(True)
        self._filter_combo.clear()
        self._filter_combo.addItem("All classes")
        for cls in sorted(self._known_classes):
            self._filter_combo.addItem(cls)
        # Restore selection if the previously-filtered class is still present
        if prev:
            idx = self._filter_combo.findText(prev)
            if idx >= 0:
                self._filter_combo.setCurrentIndex(idx)
        self._filter_combo.blockSignals(False)

    def _on_filter_changed(self, idx: int) -> None:
        if idx <= 0:
            self._filter_class = ""
        else:
            self._filter_class = self._filter_combo.currentText().lower()

    # ---------------------------------------------------------------
    # Overlay drawing — same look as the small camera tab, but bigger
    # ---------------------------------------------------------------

    @staticmethod
    def _draw_overlays(frame: np.ndarray, detections: list) -> None:
        h, w = frame.shape[:2]
        if not detections:
            cv2.putText(
                frame, "NO DETECTIONS",
                (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
            return
        # Box thickness and font scale up with frame size
        box_thick = max(2, w // 320)
        font_scale = 0.45 + (w / 1600.0)
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            colour = (255, 99, 108)        # BGR of accent purple
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thick)

            label = f"{det.class_name} {det.confidence:.2f}"
            if det.distance_m and det.distance_m > 0:
                label += f"  {det.distance_m:.2f}m"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
            )
            pad = 6
            cv2.rectangle(
                frame, (x1, y1 - th - pad * 2),
                (x1 + tw + pad * 2, y1), colour, -1,
            )
            cv2.putText(
                frame, label, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (240, 240, 245), 1, cv2.LINE_AA,
            )
