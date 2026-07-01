"""
Skill Config Dialog — edit per-skill parameters from skills_config.yaml.

Left-side skill list, right-side parameter form. Loads and saves to
``config/skills_config.yaml``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Tokens

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "skills_config.yaml",
)


class SkillConfigDialog(QDialog):
    """Modal editor for config/skills_config.yaml skill parameters."""

    def __init__(self, skills_config: Dict[str, Any], *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Skill Configuration")
        self.setMinimumSize(620, 480)
        self._config = _deep_copy(skills_config)
        self._pages: Dict[str, Dict[str, Any]] = {}          # skill -> {param: widget}
        self._build()

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)

        body = QHBoxLayout()
        body.setSpacing(12)

        # Left: skill list
        self._skill_list = QListWidget()
        self._skill_list.setMinimumWidth(160)
        self._skill_list.setMaximumWidth(200)
        for name in sorted(self._config.keys()):
            item = QListWidgetItem(name.replace("_", " ").title())
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._skill_list.addItem(item)
        self._skill_list.currentRowChanged.connect(self._on_skill_changed)
        body.addWidget(self._skill_list)

        # Right: stacked parameter forms
        self._stack = QStackedWidget()
        for idx, name in enumerate(sorted(self._config.keys())):
            page = self._build_param_page(name, self._config.get(name, {}))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            self._stack.addWidget(scroll)
        body.addWidget(self._stack, 1)

        outer.addLayout(body, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Select first skill
        if self._skill_list.count() > 0:
            self._skill_list.setCurrentRow(0)

    def _build_param_page(self, skill_name: str, params: Dict[str, Any]) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)

        title = QLabel(skill_name.replace("_", " ").title())
        title.setProperty("role", "heading")
        form.addRow(title)

        widgets: Dict[str, Any] = {}
        for key, value in params.items():
            w = self._widget_for_value(key, value)
            form.addRow(self._pretty_label(key), w)
            widgets[key] = w

        self._pages[skill_name] = widgets
        return page

    # ---------------------------------------------------------------
    # Widget inference
    # ---------------------------------------------------------------

    @staticmethod
    def _widget_for_value(key: str, value: Any) -> QWidget:
        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(value)
            return cb
        if isinstance(value, int):
            sb = QSpinBox()
            sb.setRange(-9999, 99999)
            sb.setValue(value)
            return sb
        if isinstance(value, float):
            dsb = QDoubleSpinBox()
            dsb.setRange(-999.0, 999.0)
            dsb.setDecimals(3)
            dsb.setSingleStep(0.01)
            dsb.setValue(value)
            return dsb
        if isinstance(value, list):
            le = QLineEdit(str(value))
            le.setPlaceholderText("e.g. [0, 0, 0]")
            return le
        if isinstance(value, str) and "|" in key:
            cb = QComboBox()
            cb.addItems([v.strip() for v in value.split("|") if v.strip()])
            cb.setCurrentText(value.split("|")[0].strip())
            return cb
        # default: line edit
        le = QLineEdit(str(value))
        return le

    @staticmethod
    def _pretty_label(key: str) -> str:
        return key.replace("_", " ").title()

    # ---------------------------------------------------------------
    # Slot
    # ---------------------------------------------------------------

    def _on_skill_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    # ---------------------------------------------------------------
    # Collect & save
    # ---------------------------------------------------------------

    def _collect(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for skill_name, widgets in self._pages.items():
            params: Dict[str, Any] = {}
            original = self._config.get(skill_name, {})
            for key, widget in widgets.items():
                orig_val = original.get(key)
                if isinstance(widget, QCheckBox):
                    params[key] = widget.isChecked()
                elif isinstance(widget, QSpinBox):
                    params[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[key] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[key] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    # Try to parse as the original type
                    if isinstance(orig_val, list):
                        try:
                            import ast
                            params[key] = ast.literal_eval(text)
                        except Exception:
                            params[key] = text
                    elif isinstance(orig_val, (int, float)):
                        try:
                            params[key] = type(orig_val)(text)
                        except (ValueError, TypeError):
                            params[key] = text
                    else:
                        params[key] = text
            out[skill_name] = params
        return out

    def _on_save(self) -> None:
        collected = self._collect()
        try:
            with open(_CONFIG_PATH, "w") as f:
                yaml.safe_dump(collected, f, default_flow_style=False, sort_keys=False)
            logger.info("Skills config saved to %s", _CONFIG_PATH)
        except Exception as exc:
            logger.error("Failed to save skills config: %s", exc)
        self.accept()

    def get_config(self) -> Dict[str, Any]:
        """Return the current (possibly edited) config."""
        return self._collect()


def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    import json
    return json.loads(json.dumps(d))
