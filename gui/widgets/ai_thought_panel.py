"""
AIThoughtPanelWidget — VLM scene analysis on the left, LLM reasoning on the
right. Reads SharedState.vlm_last_output / llm_last_output and refreshes when
the AI mode emits its task signals or when the underlying objects change.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens


def _section_card(title: str, hint: str = "") -> tuple:
    """Return (frame, body QPlainTextEdit) — a card-styled container."""
    card = QFrame()
    card.setProperty("role", "card")
    v = QVBoxLayout(card)
    v.setContentsMargins(14, 14, 14, 14)
    v.setSpacing(8)

    cap = QLabel(title)
    cap.setProperty("role", "caption")
    v.addWidget(cap)

    if hint:
        h = QLabel(hint)
        h.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 11px;")
        h.setWordWrap(True)
        v.addWidget(h)

    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setProperty("role", "log")
    body.setPlaceholderText("(empty)")
    v.addWidget(body, 1)
    return card, body


class AIThoughtPanelWidget(QWidget):
    def __init__(
        self,
        state: SharedState,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state

        self._vlm_card, self._vlm_text = _section_card(
            "VLM SCENE ANALYSIS",
            "What the vision model perceives — scene description, "
            "object relationships, navigation hints.",
        )
        self._llm_card, self._llm_text = _section_card(
            "LLM REASONING",
            "Step-by-step planner thinking, action decision, "
            "confidence and next observation.",
        )

        # Header strip with model + latency badges
        self._vlm_meta = QLabel("")
        self._vlm_meta.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-family: 'JetBrains Mono'; font-size: 11px;"
        )
        vlm_layout = self._vlm_card.layout()
        vlm_layout.insertWidget(2, self._vlm_meta)

        self._llm_meta = QLabel("")
        self._llm_meta.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-family: 'JetBrains Mono'; font-size: 11px;"
        )
        llm_layout = self._llm_card.layout()
        llm_layout.insertWidget(2, self._llm_meta)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.addWidget(self._vlm_card, 1)
        outer.addWidget(self._llm_card, 1)

        # ROS bridge / mode signals push into shared state; we just poll periodically.
        # Paused via showEvent/hideEvent so the panel doesn't poll when invisible.
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._refresh)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
            self._refresh()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    # ---------------------------------------------------------------

    def _refresh(self) -> None:
        with self.state.lock:
            vlm = self.state.vlm_last_output
            llm = self.state.llm_last_output

        if vlm is not None:
            sections = []
            if vlm.scene_description:
                sections.append("SCENE\n" + vlm.scene_description)
            if vlm.object_relationships:
                sections.append("OBJECTS\n" + vlm.object_relationships)
            if vlm.navigation_hints:
                sections.append("NAVIGATION\n" + vlm.navigation_hints)
            if vlm.task_observations:
                sections.append("TASK\n" + vlm.task_observations)
            text = "\n\n".join(sections) or vlm.raw_text
            self._set_if_changed(self._vlm_text, text)
            self._vlm_meta.setText(
                f"{vlm.model}   ·   {vlm.latency_ms:.0f} ms   ·   {vlm.tokens_used} tok"
            )

        if llm is not None:
            parts = []
            parts.append(f"STATUS: {llm.status or '—'}")
            parts.append(f"CONFIDENCE: {llm.confidence:.2f}")
            parts.append("")
            parts.append("REASONING")
            parts.append(llm.reasoning or "(empty)")
            if llm.action_type == "low_level" and llm.low_level_command:
                cmd = llm.low_level_command
                parts.append("")
                parts.append(
                    f"ACTION (low_level): linear={cmd.get('linear_x', 0.0):.2f} "
                    f"angular={cmd.get('angular_z', 0.0):.2f} "
                    f"duration={cmd.get('duration_ms', 0)} ms"
                )
            elif llm.action_type == "skill" and llm.skill_command:
                parts.append("")
                parts.append(
                    f"ACTION (skill): {llm.skill_command.get('skill_name', '?')}"
                )
                params = llm.skill_command.get("parameters") or {}
                if params:
                    parts.append(f"  parameters: {params}")
            if llm.next_observation:
                parts.append("")
                parts.append("NEXT")
                parts.append(llm.next_observation)
            text = "\n".join(parts)
            self._set_if_changed(self._llm_text, text)
            self._llm_meta.setText(
                f"{llm.model}   ·   {llm.latency_ms:.0f} ms   ·   {llm.tokens_used} tok"
            )

    @staticmethod
    def _set_if_changed(box: QPlainTextEdit, text: str) -> None:
        # Avoid re-setting identical text — keeps cursor position stable
        if box.toPlainText() == text:
            return
        # Preserve scroll position if user scrolled
        sb = box.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 8
        box.setPlainText(text)
        if at_bottom:
            sb.setValue(sb.maximum())
