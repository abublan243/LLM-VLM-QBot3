"""
Settings Dialog — modal editor for config/settings.json with API key validation.

Grouped by config section (Robot, API Keys, AI Defaults, Safety, UI, Simulation,
Logging). API key fields have one-click validate buttons that call
``model_registry.validate_openai_key()`` / ``validate_google_key()``.

Emits ``settings_changed(dict)`` on save.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.theme import StatusPill

logger = logging.getLogger(__name__)

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json",
)


class SettingsDialog(QDialog):
    """Modal settings.json editor with live API key validation."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, settings: Dict[str, Any], *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(560, 640)
        self._settings = _deep_copy(settings)
        self._widgets: Dict[str, Any] = {}
        self._build()

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        form = QVBoxLayout(container)
        form.setSpacing(14)

        form.addWidget(self._group_robot())
        form.addWidget(self._group_api_keys())
        form.addWidget(self._group_ai_defaults())
        form.addWidget(self._group_calibration())
        form.addWidget(self._group_safety())
        form.addWidget(self._group_ui())
        form.addWidget(self._group_simulation())
        form.addWidget(self._group_logging())
        form.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    # ---- Section builders ----

    def _group_robot(self) -> QGroupBox:
        g = QGroupBox("Robot Connection (Native DDS)")
        f = QFormLayout(g)
        self._widgets["robot.ros_domain_id"] = self._spin(
            "robot", "ros_domain_id", f, "ROS Domain ID", 0, 232,
        )
        rmw = QComboBox()
        rmw.addItems([
            "rmw_cyclonedds_cpp",
            "rmw_fastrtps_cpp",
            "rmw_connextdds",
        ])
        rmw.setCurrentText(str(self._settings.get("robot", {}).get(
            "rmw_implementation", "rmw_cyclonedds_cpp")))
        f.addRow("RMW Implementation", rmw)
        self._widgets["robot.rmw_implementation"] = rmw

        self._widgets["robot.node_name"] = self._line(
            "robot", "node_name", f, "Host Node Name",
        )
        return g

    def _group_api_keys(self) -> QGroupBox:
        g = QGroupBox("API Keys")
        v = QVBoxLayout(g)

        # OpenAI
        row_oai = QHBoxLayout()
        oai_edit = QLineEdit(str(self._settings.get("api_keys", {}).get("openai", "")))
        oai_edit.setEchoMode(QLineEdit.EchoMode.Password)
        oai_edit.setPlaceholderText("sk-…")
        row_oai.addWidget(QLabel("OpenAI"))
        row_oai.addWidget(oai_edit, 1)
        oai_pill = QLabel("—")
        oai_pill.setProperty("role", "pill")
        oai_pill.setMinimumWidth(64)
        oai_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_oai.addWidget(oai_pill)
        oai_btn = QPushButton("Validate")
        oai_btn.setProperty("variant", "ghost")
        oai_btn.clicked.connect(lambda: self._validate_key("openai", oai_edit, oai_pill))
        row_oai.addWidget(oai_btn)
        v.addLayout(row_oai)
        self._widgets["api_keys.openai"] = oai_edit

        # Google
        row_ggl = QHBoxLayout()
        ggl_edit = QLineEdit(str(self._settings.get("api_keys", {}).get("google", "")))
        ggl_edit.setEchoMode(QLineEdit.EchoMode.Password)
        ggl_edit.setPlaceholderText("AIza…")
        row_ggl.addWidget(QLabel("Google"))
        row_ggl.addWidget(ggl_edit, 1)
        ggl_pill = QLabel("—")
        ggl_pill.setProperty("role", "pill")
        ggl_pill.setMinimumWidth(64)
        ggl_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_ggl.addWidget(ggl_pill)
        ggl_btn = QPushButton("Validate")
        ggl_btn.setProperty("variant", "ghost")
        ggl_btn.clicked.connect(lambda: self._validate_key("google", ggl_edit, ggl_pill))
        row_ggl.addWidget(ggl_btn)
        v.addLayout(row_ggl)
        self._widgets["api_keys.google"] = ggl_edit

        return g

    def _group_ai_defaults(self) -> QGroupBox:
        g = QGroupBox("AI Defaults")
        f = QFormLayout(g)
        self._widgets["ai_defaults.llm_model"] = self._line("ai_defaults", "llm_model", f, "LLM Model")
        self._widgets["ai_defaults.vlm_model"] = self._line("ai_defaults", "vlm_model", f, "VLM Model")

        et = QComboBox()
        et.addItems(["high_level", "low_level"])
        et.setCurrentText(str(self._settings.get("ai_defaults", {}).get("execution_type", "high_level")))
        f.addRow("Execution Type", et)
        self._widgets["ai_defaults.execution_type"] = et

        # YOLO model size — biggest single-knob impact on GUI smoothness.
        # nano:    ~5 ms/frame on CPU,   ~5 MB weights, less accurate
        # small:   ~10 ms,               ~10 MB
        # medium:  ~25 ms,               ~25 MB
        # large:   ~50 ms,               ~50 MB  ← original default
        # x-large: ~100 ms,              ~100 MB
        ym = QComboBox()
        ym.addItems(["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
        ym.setCurrentText(str(self._settings.get("ai_defaults", {}).get("yolo_model", "yolo11l.pt")))
        f.addRow("YOLO Model", ym)
        self._widgets["ai_defaults.yolo_model"] = ym

        self._widgets["ai_defaults.yolo_confidence"] = self._dspin("ai_defaults", "yolo_confidence", f, "YOLO Confidence", 0.05, 0.99, step=0.05)
        self._widgets["ai_defaults.llm_temperature"] = self._dspin("ai_defaults", "llm_temperature", f, "LLM Temperature", 0.0, 2.0, step=0.1)
        self._widgets["ai_defaults.llm_max_tokens"] = self._spin("ai_defaults", "llm_max_tokens", f, "LLM Max Tokens", 64, 8192)
        self._widgets["ai_defaults.vlm_max_tokens"] = self._spin("ai_defaults", "vlm_max_tokens", f, "VLM Max Tokens", 64, 8192)
        return g

    def _group_calibration(self) -> QGroupBox:
        """Camera mounting + obstacle-height knobs that drive SLAM and the 3D view.

        These read straight into core.sensor_processor.transform_optical_to_base_link
        so the floor in the 3D viewer reads horizontal once camera_pitch_deg
        matches the physical RealSense tilt on the QBot3.
        """
        g = QGroupBox("Calibration (camera mounting)")
        f = QFormLayout(g)
        self._widgets["calibration.camera_height_m"] = self._dspin(
            "calibration", "camera_height_m", f,
            "Camera height (m)", 0.0, 1.0, step=0.01,
        )
        self._widgets["calibration.camera_pitch_deg"] = self._dspin(
            "calibration", "camera_pitch_deg", f,
            "Camera pitch (°, +nose-down)", -45.0, 45.0, step=0.5,
        )
        self._widgets["calibration.obstacle_min_height_m"] = self._dspin(
            "calibration", "obstacle_min_height_m", f,
            "Obstacle min height (m)", 0.0, 1.0, step=0.01,
        )
        self._widgets["calibration.obstacle_max_height_m"] = self._dspin(
            "calibration", "obstacle_max_height_m", f,
            "Obstacle max height (m)", 0.1, 3.0, step=0.05,
        )

        # Gyro-calibration toggles. The Pi node reads the matching ROS2
        # parameter at launch time, so this is informational here — to
        # actually disable the 5 s wait you launch the Pi node with
        #     ros2 run qbot3 qbot3_base --ros-args -p enable_gyro_calibration:=false
        # The Skip button on the host banner is the runtime override.
        cb = QCheckBox()
        cb.setChecked(bool(
            self._settings.get("calibration", {}).get("enable_gyro_calibration", True)
        ))
        cb.setToolTip(
            "Default for the Pi-side gyro-bias calibration window.\n"
            "Takes effect on the NEXT Pi launch (use the launch arg\n"
            "  --ros-args -p enable_gyro_calibration:=true|false ).\n"
            "For runtime override, press Skip on the calibration banner."
        )
        f.addRow("Enable Gyro Calibration", cb)
        self._widgets["calibration.enable_gyro_calibration"] = cb

        self._widgets["calibration.gyro_calibration_duration_s"] = self._dspin(
            "calibration", "gyro_calibration_duration_s", f,
            "Calibration duration (s)", 1.0, 30.0, step=0.5,
        )
        return g

    def _group_safety(self) -> QGroupBox:
        g = QGroupBox("Safety")
        f = QFormLayout(g)
        self._widgets["safety.max_linear_speed"] = self._dspin("safety", "max_linear_speed", f, "Max Linear (m/s)", 0.05, 1.0, step=0.05)
        self._widgets["safety.max_angular_speed"] = self._dspin("safety", "max_angular_speed", f, "Max Angular (rad/s)", 0.1, 3.0, step=0.1)
        self._widgets["safety.min_battery_percent"] = self._spin("safety", "min_battery_percent", f, "Min Battery %", 5, 50)
        self._widgets["safety.obstacle_stop_distance_m"] = self._dspin("safety", "obstacle_stop_distance_m", f, "Obstacle Stop (m)", 0.05, 1.0, step=0.05)
        return g

    def _group_ui(self) -> QGroupBox:
        g = QGroupBox("UI")
        f = QFormLayout(g)
        self._widgets["ui.camera_display_fps"] = self._spin("ui", "camera_display_fps", f, "Camera FPS", 1, 60)
        self._widgets["ui.plot_history_seconds"] = self._spin("ui", "plot_history_seconds", f, "Plot History (s)", 10, 600)

        dm = QComboBox()
        dm.addItems(["manual", "ai", "skills"])
        dm.setCurrentText(str(self._settings.get("ui", {}).get("default_mode", "manual")))
        f.addRow("Default Mode", dm)
        self._widgets["ui.default_mode"] = dm
        return g

    def _group_simulation(self) -> QGroupBox:
        g = QGroupBox("Simulation")
        f = QFormLayout(g)
        cb = QCheckBox()
        cb.setChecked(bool(self._settings.get("simulation", {}).get("enable_when_disconnected", True)))
        f.addRow("Enable When Disconnected", cb)
        self._widgets["simulation.enable_when_disconnected"] = cb
        self._widgets["simulation.synthetic_camera_fps"] = self._spin("simulation", "synthetic_camera_fps", f, "Synth Camera FPS", 1, 30)
        return g

    def _group_logging(self) -> QGroupBox:
        g = QGroupBox("Logging")
        f = QFormLayout(g)
        lv = QComboBox()
        lv.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        lv.setCurrentText(str(self._settings.get("logging", {}).get("level", "INFO")))
        f.addRow("Level", lv)
        self._widgets["logging.level"] = lv
        self._widgets["logging.file"] = self._line("logging", "file", f, "Log File")
        return g

    # ---- Widget factories ----

    def _line(self, section: str, key: str, form: QFormLayout, label: str) -> QLineEdit:
        w = QLineEdit(str(self._settings.get(section, {}).get(key, "")))
        form.addRow(label, w)
        return w

    def _spin(self, section: str, key: str, form: QFormLayout, label: str,
              lo: int, hi: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(self._settings.get(section, {}).get(key, lo)))
        form.addRow(label, w)
        return w

    def _dspin(self, section: str, key: str, form: QFormLayout, label: str,
               lo: float, hi: float, *, step: float = 0.01) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(3 if step < 0.01 else 2)
        w.setValue(float(self._settings.get(section, {}).get(key, lo)))
        form.addRow(label, w)
        return w

    # ---------------------------------------------------------------
    # API key validation
    # ---------------------------------------------------------------

    def _validate_key(self, provider: str, edit: QLineEdit, pill: QLabel) -> None:
        key = edit.text().strip()
        if not key:
            StatusPill.set_tone(pill, "warn")
            pill.setText("empty")
            return
        try:
            from ai.model_registry import validate_openai_key, validate_google_key
            if provider == "openai":
                ok, message = validate_openai_key(key)
            else:
                ok, message = validate_google_key(key)
            if ok:
                StatusPill.set_tone(pill, "ok")
                pill.setText("valid")
                pill.setToolTip(message)
            else:
                StatusPill.set_tone(pill, "bad")
                pill.setText("invalid")
                pill.setToolTip(message)
        except Exception as exc:
            StatusPill.set_tone(pill, "bad")
            pill.setText("error")
            pill.setToolTip(str(exc))
            logger.warning("Key validation failed: %s", exc)

    # ---------------------------------------------------------------
    # Save / collect
    # ---------------------------------------------------------------

    def _collect(self) -> Dict[str, Any]:
        """Collect widget values back into the settings dict."""
        s = _deep_copy(self._settings)
        for dotpath, widget in self._widgets.items():
            section, key = dotpath.split(".", 1)
            if section not in s:
                s[section] = {}
            if isinstance(widget, QLineEdit):
                s[section][key] = widget.text()
            elif isinstance(widget, QSpinBox):
                s[section][key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                s[section][key] = widget.value()
            elif isinstance(widget, QComboBox):
                s[section][key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                s[section][key] = widget.isChecked()
        return s

    def _on_save(self) -> None:
        new_settings = self._collect()
        try:
            with open(_SETTINGS_PATH, "w") as f:
                json.dump(new_settings, f, indent=2)
            logger.info("Settings saved to %s", _SETTINGS_PATH)
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)
        self.settings_changed.emit(new_settings)
        self.accept()


def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(d))
