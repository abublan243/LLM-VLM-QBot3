"""
Main Window — top-level QMainWindow that wires all widgets, modes, and signals.

Layout:
    ┌──────────── 65 % ────────────┬──── 35 % ────┐
    │   QTabWidget                 │ Mode selector │
    │   ├ Camera / SLAM / 3D       │ (active panel)│
    │   ├ Sensors                  │               │
    │   ├ AI Thought               │               │
    │   └ Performance              │               │
    ├──────────────────────────────┴───────────────┤
    │                 Status bar                    │
    └──────────────────────────────────────────────┘

Keyboard shortcuts:
    W/A/S/D, Arrows      Teleop (manual mode)
    Space                 Emergency stop
    Esc                   Cancel AI task
    1 / 2 / 3             Switch mode (AI / Manual / Skills)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.theme import ModeButton, StatusPill, Tokens

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Application shell — modes, widgets, and signals fully wired."""

    def __init__(
        self,
        state,
        ros_bridge,
        perf_monitor,
        slam_manager,
        vlm_pipeline,
        llm_planner,
        mode_manual,
        mode_skills,
        mode_ai,
        *,
        settings: Optional[Dict[str, Any]] = None,
        skills_config: Optional[Dict[str, Any]] = None,
        voice_io: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros_bridge
        self.perf = perf_monitor
        self.slam = slam_manager
        self.vlm = vlm_pipeline
        self.llm = llm_planner
        self.mode_manual = mode_manual
        self.mode_skills = mode_skills
        self.mode_ai = mode_ai
        self.voice = voice_io
        self._settings = settings or {}
        self._skills_config = skills_config or {}

        self._active_mode = "manual"
        self._keys_held: set = set()

        self.setWindowTitle("QBot3 — Autonomous Inspection & Assistance")
        ui = self._settings.get("ui", {})
        self.setMinimumSize(
            ui.get("window_min_width", 1280),
            ui.get("window_min_height", 720),
        )
        self.resize(
            ui.get("window_default_width", 1920),
            ui.get("window_default_height", 1080),
        )

        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self._wire_signals()

        # Start in default mode
        self._switch_mode(ui.get("default_mode", "manual"))

    # ================================================================
    # Build
    # ================================================================

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        file_menu.addAction("Settings…", self._open_settings)
        file_menu.addAction("Skills Config…", self._open_skills_config)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        help_menu = bar.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Top-level vertical: optional calibration banner (auto-hidden
        # when calibrated) on top, then the 65/35 horizontal splitter below.
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._calib_banner = self._build_calibration_banner()
        outer.addWidget(self._calib_banner)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        outer.addWidget(splitter, 1)

        # ---- Left: tab viewer (65 %) ----
        self._tabs = QTabWidget()

        # Vision tab — full-size RGB + YOLO. The operator can land here to
        # see the camera + detector at full size without drilling into the
        # smaller Camera card's RGB sub-tab. The Live (pure-RGB, event-driven)
        # view is now a sub-tab inside Camera, alongside RGB / Depth / SLAM / 3D.
        from gui.widgets.vision_widget import VisionWidget
        self._vision = VisionWidget(
            self.state, vlm_pipeline=self.vlm, parent=self,
        )
        # Toggles persist via the MainWindow — settings get re-saved so the
        # operator's per-layer choice survives an app restart.
        self._vision.persist_layer_change = self._persist_layer_toggle
        self._tabs.addTab(self._vision, "Vision")

        from gui.widgets.camera_viewer import CameraViewerWidget
        cal = self._settings.get("calibration", {})
        self._camera_viewer = CameraViewerWidget(
            self.state, self.ros, self.slam,
            camera_height_m=cal.get("camera_height_m", 0.10),
            camera_pitch_deg=cal.get("camera_pitch_deg", 0.0),
            parent=self,
        )
        self._tabs.addTab(self._camera_viewer, "Camera")

        from gui.widgets.sensor_panel import SensorPanelWidget
        self._sensor_panel = SensorPanelWidget(self.state, self.ros, parent=self)
        self._tabs.addTab(self._sensor_panel, "Sensors")

        from gui.widgets.ai_thought_panel import AIThoughtPanelWidget
        self._ai_thought = AIThoughtPanelWidget(self.state, parent=self)
        self._tabs.addTab(self._ai_thought, "AI Thought")

        from gui.widgets.performance_panel import PerformancePanelWidget
        self._perf_panel = PerformancePanelWidget(parent=self)
        self._tabs.addTab(self._perf_panel, "Performance")

        # Monitor — operator-configurable 2×2 grid that combines any
        # subset of the above streams in a single view.
        from gui.widgets.monitor_widget import MonitorWidget
        self._monitor = MonitorWidget(
            self.state, self.slam, self.perf, parent=self,
        )
        self._tabs.addTab(self._monitor, "Monitor")

        splitter.addWidget(self._tabs)

        # ---- Right: mode selector + active panel (35 %) ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        # Mode buttons
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._btn_ai = ModeButton("AI")
        self._btn_manual = ModeButton("Manual")
        self._btn_skills = ModeButton("Skills")
        for btn, mode in [(self._btn_ai, "ai"), (self._btn_manual, "manual"),
                          (self._btn_skills, "skills")]:
            btn.setMinimumHeight(42)
            btn.clicked.connect(lambda checked, m=mode: self._switch_mode(m))
            mode_row.addWidget(btn)

        # E-stop
        from PyQt6.QtWidgets import QPushButton
        estop = QPushButton("E-STOP")
        estop.setProperty("variant", "estop")
        estop.setMinimumHeight(42)
        estop.clicked.connect(self._emergency_stop)
        mode_row.addWidget(estop)

        rv.addLayout(mode_row)

        # Stacked panels
        self._panel_stack = QStackedWidget()

        from gui.widgets.control_panel import ControlPanelWidget
        self._control_panel = ControlPanelWidget(
            voice_io=self.voice,
            mode_ai=self.mode_ai,
            parent=self,
        )
        self._panel_stack.addWidget(self._control_panel)       # index 0 = AI

        from gui.widgets.manual_control import ManualControlWidget
        self._manual_panel = ManualControlWidget(self.state, self.mode_manual, parent=self)
        self._panel_stack.addWidget(self._manual_panel)        # index 1 = Manual

        # Skills panel — skill cards + run/abort
        self._skills_panel = self._build_skills_panel()
        self._panel_stack.addWidget(self._skills_panel)        # index 2 = Skills

        rv.addWidget(self._panel_stack, 1)
        splitter.addWidget(right)

        # Splitter proportions: 65 / 35
        splitter.setStretchFactor(0, 65)
        splitter.setStretchFactor(1, 35)

    def _build_skills_panel(self) -> QWidget:
        """Mode-3 panel — Mindstorms-EV3-style drag-drop block programming."""
        from gui.widgets.block_programming import BlockProgrammingWidget

        ai_defaults = self._settings.get("ai_defaults", {})
        widget = BlockProgrammingWidget(
            self.state, self.ros,
            vlm_pipeline=self.vlm,
            vlm_model_name=ai_defaults.get("vlm_model", "gpt-4o"),
            skills_config=self._skills_config,
            voice_io=self.voice,
            parent=self,
        )
        widget.program_started.connect(
            lambda: self._status_label.setText("Block program started")
        )
        widget.program_finished.connect(
            lambda ok, msg: self._status_label.setText(
                ("Program: " + msg) if ok else ("Program stopped: " + msg)
            )
        )
        self._block_panel = widget
        return widget

    def _build_calibration_banner(self) -> QWidget:
        """Top banner that pulses gold while the Pi is calibrating the gyro,
        then briefly turns green for a "calibration done" confirmation, then
        hides itself. Watches ros_bridge.imu_calibration_changed.
        """
        banner = QWidget()
        banner.setObjectName("calibBanner")
        banner.setStyleSheet(
            f"QWidget#calibBanner {{ "
            f"  background-color: rgba(245, 158, 11, 38); "
            f"  border: 1px solid {Tokens.WARNING}; "
            f"  border-radius: {Tokens.RADIUS_MD}px; "
            f"}}"
        )
        h = QHBoxLayout(banner)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(12)

        self._calib_dot = QLabel("●")
        self._calib_dot.setStyleSheet(
            f"color: {Tokens.WARNING}; font-size: 14px;"
        )
        h.addWidget(self._calib_dot)

        self._calib_text = QLabel("Calibrating gyroscope — keep robot still")
        self._calib_text.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY}; font-weight: 600;"
        )
        h.addWidget(self._calib_text)

        h.addStretch(1)

        # A simple percent readout — pyqtgraph progress bars are heavy
        self._calib_pct = QLabel("0 %")
        self._calib_pct.setStyleSheet(
            f"color: {Tokens.WARNING}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 12px; font-weight: 700;"
        )
        h.addWidget(self._calib_pct)

        # Skip button — operator can abort the calibration window early.
        # The Pi will lock in whatever samples it has so far (or bias=0 if
        # the window had just started — online refinement takes over).
        self._calib_skip_btn = QPushButton("Skip")
        self._calib_skip_btn.setProperty("variant", "ghost")
        self._calib_skip_btn.setToolTip(
            "Abort the gyro-calibration window now.\n"
            "Bias is locked in from whatever samples have been gathered "
            "(could be 0 if pressed immediately after launch).\n"
            "To skip calibration entirely on every Pi restart, launch with:\n"
            "    ros2 run qbot3 qbot3_base --ros-args -p enable_gyro_calibration:=false"
        )
        self._calib_skip_btn.clicked.connect(self._on_skip_calibration)
        h.addWidget(self._calib_skip_btn)

        # Hide until we know we're calibrating (i.e. first message arrives).
        # If the Pi never publishes /qbot3/imu/calibrated, the banner just
        # stays hidden and the operator can still drive — no harm.
        banner.hide()
        # Latches True after the green "ready" confirmation has been shown
        # once, so subsequent calibrated=True messages (Pi publishes at 5 Hz)
        # don't keep re-showing the banner.
        self._calib_done_shown = False
        return banner

    def _on_skip_calibration(self) -> None:
        """Banner Skip button → tell the Pi to finalise the window now."""
        if hasattr(self.ros, "publish_skip_calibration"):
            ok = self.ros.publish_skip_calibration()
            if ok:
                self._status_label.setText("Skip requested — Pi will finalise calibration")
            else:
                self._status_label.setText("Skip failed — bridge not connected?")
        else:
            self._status_label.setText("Skip not supported by this bridge")

    def _on_imu_calibration_changed(self, calibrated: bool, progress: float) -> None:
        """Slot wired to ROSBridge.imu_calibration_changed."""
        if calibrated:
            # The Pi publishes calibrated=True at 5 Hz. Only react to the
            # first one — otherwise show() is re-called every 200 ms and
            # the banner never hides.
            if self._calib_done_shown:
                return
            self._calib_done_shown = True
            self._calib_dot.setStyleSheet(
                f"color: {Tokens.SUCCESS}; font-size: 14px;"
            )
            self._calib_text.setText("Gyroscope calibrated — ready")
            self._calib_pct.setText("100 %")
            self._calib_pct.setStyleSheet(
                f"color: {Tokens.SUCCESS}; "
                f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 12px; font-weight: 700;"
            )
            # Hide Skip — calibration is already done
            self._calib_skip_btn.hide()
            self._calib_banner.setStyleSheet(
                f"QWidget#calibBanner {{ "
                f"  background-color: rgba(34, 197, 94, 38); "
                f"  border: 1px solid {Tokens.SUCCESS}; "
                f"  border-radius: {Tokens.RADIUS_MD}px; "
                f"}}"
            )
            self._calib_banner.show()
            # Auto-hide the green confirmation after 2 s
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, self._calib_banner.hide)
        else:
            # Re-arm the latch so a re-triggered calibration (Reset
            # map+odom+gyro) shows the green confirmation again.
            self._calib_done_shown = False
            pct = int(round(max(0.0, min(1.0, progress)) * 100))
            self._calib_text.setText(
                f"Calibrating gyroscope — keep robot still ({5 - int(progress * 5)} s)"
            )
            self._calib_pct.setText(f"{pct} %")
            self._calib_pct.setStyleSheet(
                f"color: {Tokens.WARNING}; "
                f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 12px; font-weight: 700;"
            )
            self._calib_dot.setStyleSheet(
                f"color: {Tokens.WARNING}; font-size: 14px;"
            )
            # Skip is only meaningful while calibrating
            self._calib_skip_btn.show()
            self._calib_banner.setStyleSheet(
                f"QWidget#calibBanner {{ "
                f"  background-color: rgba(245, 158, 11, 38); "
                f"  border: 1px solid {Tokens.WARNING}; "
                f"  border-radius: {Tokens.RADIUS_MD}px; "
                f"}}"
            )
            self._calib_banner.show()

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._conn_pill = QLabel("DISCONNECTED")
        self._conn_pill.setProperty("role", "pill")
        StatusPill.set_tone(self._conn_pill, "bad")
        sb.addPermanentWidget(self._conn_pill)

        self._batt_pill = QLabel("BATT —")
        self._batt_pill.setProperty("role", "pill")
        sb.addPermanentWidget(self._batt_pill)

        self._mode_label = QLabel("MODE: MANUAL")
        self._mode_label.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        sb.addWidget(self._mode_label)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 11px;")
        sb.addWidget(self._status_label)

    # ================================================================
    # Signal wiring
    # ================================================================

    def _wire_signals(self) -> None:
        # ROS bridge
        self.ros.connection_changed.connect(self._on_connection_changed)
        self.ros.battery_updated.connect(self._on_battery_updated)
        self.ros.imu_calibration_changed.connect(self._on_imu_calibration_changed)

        # Performance monitor → performance panel
        self.perf.metrics_updated.connect(self._perf_panel.set_metrics)

        # AI mode → control panel
        self.mode_ai.pipeline_stage.connect(self._control_panel.set_pipeline_stage)
        self.mode_ai.task_started.connect(
            lambda desc: self._control_panel.set_running(True, desc)
        )
        self.mode_ai.task_completed.connect(self._on_ai_task_completed)

        # Control panel → AI mode
        self._control_panel.run_clicked.connect(self._on_run_ai_task)
        self._control_panel.cancel_clicked.connect(self.mode_ai.cancel_task)
        self._control_panel.execution_type_changed.connect(self.mode_ai.set_execution_type)

        # Manual → save base
        self._manual_panel.save_base_clicked.connect(
            lambda: self.state.save_named_waypoint("base")
        )

    # ================================================================
    # Mode switching
    # ================================================================

    def _switch_mode(self, mode: str) -> None:
        if mode == self._active_mode:
            return

        # Deactivate current
        if self._active_mode == "manual":
            self.mode_manual.deactivate()
        elif self._active_mode == "ai":
            self.mode_ai.deactivate()
        elif self._active_mode == "skills":
            self.mode_skills.deactivate()

        self._active_mode = mode

        # Activate new
        if mode == "ai":
            self.mode_ai.activate()
            self._panel_stack.setCurrentIndex(0)
        elif mode == "manual":
            self.mode_manual.activate()
            self._panel_stack.setCurrentIndex(1)
        elif mode == "skills":
            self.mode_skills.activate()
            self._panel_stack.setCurrentIndex(2)

        # Mode button styling
        self._btn_ai.set_active(mode == "ai")
        self._btn_manual.set_active(mode == "manual")
        self._btn_skills.set_active(mode == "skills")
        self._mode_label.setText(f"MODE: {mode.upper()}")

    # ================================================================
    # Keyboard
    # ================================================================

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if event.isAutoRepeat():
            return

        # Mode switch keys
        if key == Qt.Key.Key_1:
            self._switch_mode("ai")
            return
        if key == Qt.Key.Key_2:
            self._switch_mode("manual")
            return
        if key == Qt.Key.Key_3:
            self._switch_mode("skills")
            return

        # E-stop
        if key == Qt.Key.Key_Space:
            self._emergency_stop()
            return

        # Cancel AI task
        if key == Qt.Key.Key_Escape:
            self.mode_ai.cancel_task()
            return

        # Teleop keys (manual mode)
        if self._active_mode == "manual":
            from gui.widgets.manual_control import ManualControlWidget
            km = ManualControlWidget.keymap()
            action = km.get(key)
            if action:
                self.mode_manual.press(action)
                self._keys_held.add(key)

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if event.isAutoRepeat():
            return
        if self._active_mode == "manual" and key in self._keys_held:
            from gui.widgets.manual_control import ManualControlWidget
            km = ManualControlWidget.keymap()
            action = km.get(key)
            if action:
                self.mode_manual.release(action)
            self._keys_held.discard(key)

    # ================================================================
    # Handlers
    # ================================================================

    def _emergency_stop(self) -> None:
        try:
            self.ros.publish_emergency_stop()
        except Exception:
            pass
        self.mode_manual.stop()
        self._status_label.setText("E-STOP")

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        if connected:
            StatusPill.set_tone(self._conn_pill, "ok")
            self._conn_pill.setText("CONNECTED")
        else:
            StatusPill.set_tone(self._conn_pill, "bad")
            self._conn_pill.setText(message.upper() or "DISCONNECTED")

    def _on_battery_updated(self) -> None:
        pct = self.state.battery.percent
        self._batt_pill.setText(f"BATT {pct:.0f}%")
        if pct > 50:
            StatusPill.set_tone(self._batt_pill, "ok")
        elif pct > 20:
            StatusPill.set_tone(self._batt_pill, "warn")
        else:
            StatusPill.set_tone(self._batt_pill, "bad")

    def _on_run_ai_task(self, task: str, llm_model: str, vlm_model: str,
                        exec_type: str) -> None:
        self.mode_ai.set_models(llm_model, vlm_model)
        self.mode_ai.set_execution_type(exec_type)
        self.mode_ai.submit_task(task)

    def _on_ai_task_completed(self, success: bool, message: str) -> None:
        self._control_panel.set_running(False, "")
        tag = "✓" if success else "✗"
        self._control_panel.append_task_log(f"{tag} {message}")
        self._status_label.setText(f"AI task: {tag} {message[:80]}")

    # ================================================================
    # Menu actions
    # ================================================================

    def _open_settings(self) -> None:
        from gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._settings, parent=self)
        dlg.settings_changed.connect(self._apply_settings)
        dlg.exec()

    def _apply_settings(self, new_settings: Dict[str, Any]) -> None:
        self._settings = new_settings
        keys = new_settings.get("api_keys", {})
        self.mode_ai.set_api_keys(
            keys.get("openai", ""),
            keys.get("google", ""),
        )
        ai_defaults = new_settings.get("ai_defaults", {})
        self.mode_ai.set_yolo_confidence(ai_defaults.get("yolo_confidence", 0.45))
        # Propagate the new VLM model name to the block-programming runner so
        # `read_gauge` blocks pick the new model on the next run.
        if hasattr(self, "_block_panel"):
            self._block_panel.set_vlm(self.vlm, ai_defaults.get("vlm_model", "gpt-4o"))
        self._status_label.setText("Settings applied")

    def _open_skills_config(self) -> None:
        from gui.dialogs.skill_config_dialog import SkillConfigDialog
        dlg = SkillConfigDialog(self._skills_config, parent=self)
        if dlg.exec():
            self._skills_config = dlg.get_config()
            self._status_label.setText("Skills config updated")

    def _persist_layer_toggle(self, layer_key: str, enabled: bool) -> None:
        """Vision-tab checkbox → write back to settings.json so the operator's
        choice survives an app restart. Mirrors the layer keys used by
        VLMPipeline.set_layer_enabled.
        """
        import json
        import os

        try:
            if layer_key in ("coco", "yolo", "primary"):
                # No single boolean in settings.json gates the COCO YOLO
                # primary path — we store one under ai_defaults.
                self._settings.setdefault("ai_defaults", {})["coco_enabled"] = bool(enabled)
            elif layer_key in ("yolo_world", "yw", "world"):
                self._settings.setdefault("yolo_world", {})["enabled"] = bool(enabled)
            else:
                return

            settings_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "settings.json",
            )
            with open(settings_path, "w") as f:
                json.dump(self._settings, f, indent=2)
            self._status_label.setText(
                f"Layer '{layer_key}' {'ON' if enabled else 'OFF'} (saved)"
            )
        except Exception as exc:
            logger.exception("Persist layer toggle failed: %s", exc)
            self._status_label.setText(f"Layer toggle save FAILED: {exc}")

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "QBot3 Control",
            "QBot3 — Autonomous Inspection & Assistance Robot\n\n"
            "Graduation Project\n"
            "ROS2 Foxy  •  PyQt6  •  YOLOv11  •  GPT-4o / Gemini\n\n"
            "Built with precision and care.",
        )

    # ================================================================
    # Cleanup
    # ================================================================

    def closeEvent(self, event) -> None:
        self.mode_manual.deactivate()
        self.mode_ai.deactivate()
        self.mode_skills.deactivate()
        super().closeEvent(event)
