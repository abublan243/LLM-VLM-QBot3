"""
ControlPanelWidget — the AI Mode (Mode 1) right-hand panel.

Contents (top → bottom):
    * Task input (multi-line) with example-task quick-pick combo
    * Voice row: push-to-talk mic + "Speak responses" toggle
    * Execution-type toggle (low_level vs high_level)
    * Model selectors (LLM, VLM)
    * Run / Cancel buttons
    * Pipeline status bar (5 stage chips)
    * Current task display
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.model_registry import llm_model_names, vlm_model_names
from ai.prompt_templates import EXAMPLE_TASKS, EXAMPLE_TASKS_BY_LEVEL
from gui.theme import Spinner, Tokens

logger = logging.getLogger(__name__)


class _MicButton(QPushButton):
    """Round mic button with a vector-drawn icon (no font/emoji dependency).

    Three visual states:
        * disabled   — muted grey, "mic off" diagonal slash
        * idle       — purple-tinted ring, mic glyph
        * recording  — solid red fill, white mic glyph + soft halo

    Painted by hand so the icon renders identically across systems
    (the 🎙 emoji has spotty rendering at small button sizes).
    """

    SIZE_PX = 36

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(self.SIZE_PX, self.SIZE_PX)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Strip stylesheet inheritance — we paint the whole thing ourselves.
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self.setFlat(True)

    def paintEvent(self, event) -> None:                 # noqa: D401, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = h = self.SIZE_PX
        cx, cy = w / 2.0, h / 2.0

        recording = self.isChecked() and self.isEnabled()
        enabled = self.isEnabled()

        # Background disc
        if recording:
            disc = QColor(Tokens.DANGER)
        elif enabled:
            disc = QColor(108, 99, 255, 56)              # accent-tinted
        else:
            disc = QColor(60, 60, 70, 80)                # muted

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(disc)
        p.drawEllipse(QRectF(2, 2, w - 4, h - 4))

        # Outline ring
        ring_col = QColor(Tokens.ACCENT_PRIMARY) if (enabled and not recording) else (
            QColor(Tokens.DANGER) if recording else QColor(Tokens.BORDER)
        )
        ring_col.setAlpha(220 if recording else 200)
        p.setPen(QPen(ring_col, 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(2, 2, w - 4, h - 4))

        # Mic glyph (vector). Capsule body + stand + base bar.
        glyph_col = QColor(255, 255, 255) if recording else QColor(
            Tokens.TEXT_PRIMARY if enabled else Tokens.TEXT_MUTED
        )
        p.setPen(QPen(glyph_col, 1.8))
        p.setBrush(glyph_col)
        # Capsule (rounded rect): width 9, height 13, centered, lifted slightly
        cap_w, cap_h = 9.0, 13.0
        cap_rect = QRectF(cx - cap_w / 2, cy - cap_h / 2 - 2.0, cap_w, cap_h)
        p.drawRoundedRect(cap_rect, cap_w / 2, cap_w / 2)
        # Arc/stand below the capsule (open semicircle)
        p.setBrush(Qt.BrushStyle.NoBrush)
        arc_rect = QRectF(cx - 7.0, cy - 1.5, 14.0, 12.0)
        # Qt angles are in 1/16 degrees; 200..340 deg sweep
        p.drawArc(arc_rect, int(200 * 16), int(140 * 16))
        # Vertical stand
        p.drawLine(QPointF(cx, cy + 9.0), QPointF(cx, cy + 13.5))
        # Horizontal base
        p.drawLine(QPointF(cx - 4.0, cy + 13.5), QPointF(cx + 4.0, cy + 13.5))

        # Disabled-state slash
        if not enabled:
            slash = QColor(Tokens.TEXT_MUTED)
            slash.setAlpha(180)
            p.setPen(QPen(slash, 2.0))
            p.drawLine(QPointF(7, h - 7), QPointF(w - 7, 7))

        p.end()


class _PromptWithMic(QTextEdit):
    """QTextEdit that hosts a small mic button overlaid in its bottom-right
    corner (modern composer pattern — out of the way of typed text).

    The mic is added later via `set_mic_button()`. Once added, the button
    is repositioned on every resize so it stays anchored.
    """

    _MIC_MARGIN_PX = 8

    def __init__(self, *, parent_panel: Optional[QWidget] = None) -> None:
        super().__init__(parent_panel)
        self._mic_btn: Optional[QPushButton] = None

    def set_mic_button(self, btn: QPushButton) -> None:
        self._mic_btn = btn
        # Reserve right-padding inside the viewport so typed text doesn't
        # slide under the mic button when the cursor is at the bottom row.
        if btn is not None:
            self.setViewportMargins(0, 0, btn.width() + self._MIC_MARGIN_PX * 2, 0)

    def reposition_mic(self) -> None:
        if self._mic_btn is None:
            return
        vp = self.viewport()
        if vp is None:
            return
        # Bottom-right anchor — feels organised next to a "send/run" button
        # and is the convention in ChatGPT / Claude composers.
        x = vp.width() - self._mic_btn.width() - self._MIC_MARGIN_PX
        y = vp.height() - self._mic_btn.height() - self._MIC_MARGIN_PX
        self._mic_btn.move(max(0, x), max(0, y))
        self._mic_btn.raise_()

    def resizeEvent(self, event) -> None:                # noqa: D401
        super().resizeEvent(event)
        self.reposition_mic()


PIPELINE_STAGES = ("sensor_fusion", "yolo", "vlm", "llm", "action")
STAGE_LABELS = {
    "sensor_fusion": "SENSE",
    "yolo": "YOLO",
    "vlm": "VLM",
    "llm": "LLM",
    "action": "ACT",
}


class ControlPanelWidget(QWidget):
    """Mode 1 right-side panel — emits signals for the wiring layer."""

    run_clicked = pyqtSignal(str, str, str, str)
    # task_text, llm_name, vlm_name, execution_type
    cancel_clicked = pyqtSignal()
    execution_type_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        default_llm: str = "gpt-4o-mini",
        default_vlm: str = "gpt-4o",
        default_execution: str = "high_level",
        voice_io: Optional[Any] = None,
        mode_ai: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.default_execution = default_execution
        # Voice integration is optional. Both can be None and the widget
        # still works — the mic button + speak toggle just stay hidden.
        self._voice = voice_io
        self._mode_ai = mode_ai

        self._build(default_llm, default_vlm, default_execution)

    # ---------------------------------------------------------------
    # Public API used by main_window
    # ---------------------------------------------------------------

    def set_running(self, running: bool, current_task: str = "") -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._task_input.setReadOnly(running)
        if running:
            self._spinner.start()
            self._spinner.show()
            self._current_task_label.setText(current_task or "(running)")
            self._current_task_label.setVisible(True)
        else:
            self._spinner.stop()
            self._spinner.hide()
            self._current_task_label.setVisible(False)
            self.set_pipeline_stage(None)   # clear chips

    def set_pipeline_stage(self, stage: Optional[str]) -> None:
        """Light up a single chip; pass None to clear all."""
        if stage is None:
            for s, chip in self._stage_chips.items():
                chip.setProperty("active", "false")
                chip.setProperty("done", "false")
                self._restyle(chip)
            return
        # Mark previous stages as 'done', current as 'active', later as neutral
        seen_active = False
        for s in PIPELINE_STAGES:
            chip = self._stage_chips[s]
            if s == stage:
                chip.setProperty("active", "true")
                chip.setProperty("done", "false")
                seen_active = True
            elif not seen_active:
                chip.setProperty("active", "false")
                chip.setProperty("done", "true")
            else:
                chip.setProperty("active", "false")
                chip.setProperty("done", "false")
            self._restyle(chip)

    def append_task_log(self, message: str) -> None:
        # Used by main_window when AI task completes
        self._current_task_label.setText(message)
        self._current_task_label.setVisible(True)

    # ---------------------------------------------------------------

    def _build(self, default_llm: str, default_vlm: str, default_execution: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # ---- Task input card ----
        outer.addWidget(self._build_task_card())

        # ---- Settings card ----
        outer.addWidget(self._build_settings_card(default_llm, default_vlm, default_execution))

        # ---- Run row + spinner ----
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self._run_btn = QPushButton("Run task")
        self._run_btn.setProperty("variant", "primary")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn, 1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("variant", "danger")
        self._cancel_btn.setMinimumHeight(40)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_clicked.emit)
        run_row.addWidget(self._cancel_btn)

        self._spinner = Spinner(self, size=28)
        self._spinner.hide()
        run_row.addWidget(self._spinner)
        outer.addLayout(run_row)

        # ---- Pipeline chips ----
        outer.addWidget(self._build_pipeline_card())

        # ---- Current-task display ----
        self._current_task_label = QLabel("")
        self._current_task_label.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY};"
            f"background-color: {Tokens.SURFACE_ELEVATED};"
            f"border: 1px solid {Tokens.BORDER};"
            f"border-radius: {Tokens.RADIUS_MD}px;"
            "padding: 8px 12px;"
            "font-family: 'JetBrains Mono';"
            "font-size: 11px;"
        )
        self._current_task_label.setWordWrap(True)
        self._current_task_label.setVisible(False)
        outer.addWidget(self._current_task_label)

        outer.addStretch(1)

    def _build_task_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        cap = QLabel("OPERATOR TASK")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        # Prompt input + overlaid mic button (ChatGPT-style). The mic is a
        # CHILD of the QTextEdit's viewport, repositioned to the top-right
        # corner on every resize. Click to start, click again to stop;
        # silence-detect inside VoiceIO triggers an auto-stop after the
        # operator finishes talking.
        self._task_input = _PromptWithMic(parent_panel=self)
        self._task_input.setPlaceholderText(
            "Describe the inspection or assistance task in natural language…"
        )
        self._task_input.setMinimumHeight(96)
        v.addWidget(self._task_input)

        # Inline mic state strip (small text under the input — non-intrusive)
        self._voice_status = QLabel("")
        self._voice_status.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        self._voice_status.setVisible(False)
        v.addWidget(self._voice_status)

        # Quick pick examples
        ex_row = QHBoxLayout()
        ex_row.setSpacing(8)
        ex_label = QLabel("Quick-pick:")
        ex_label.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        ex_row.addWidget(ex_label)

        self._examples = QComboBox()
        self._examples.addItem("— select an example —")
        # Round 20: present the operator with the three graduation-test
        # difficulty tiers. Group separators are non-selectable headers
        # so the operator sees explicitly which tier each task belongs
        # to during data collection.
        level_titles = {
            "easy": "─ EASY (see → approach) ─",
            "medium": "─ MEDIUM (occluded → explore → approach) ─",
            "hard": "─ HARD (sequence + return to base) ─",
        }
        for level, tasks in EXAMPLE_TASKS_BY_LEVEL.items():
            self._examples.addItem(level_titles.get(level, f"— {level} —"))
            # Disable the header item so it can't be picked as a task.
            idx = self._examples.count() - 1
            model = self._examples.model()
            try:
                model.item(idx).setEnabled(False)        # type: ignore[union-attr]
            except Exception:
                pass
            for t in tasks:
                self._examples.addItem(t)
        # A small spacer + the legacy flat list (kept for muscle memory).
        self._examples.addItem("─ misc ─")
        idx = self._examples.count() - 1
        try:
            self._examples.model().item(idx).setEnabled(False)   # type: ignore[union-attr]
        except Exception:
            pass
        for t in EXAMPLE_TASKS:
            self._examples.addItem(t)
        self._examples.currentIndexChanged.connect(self._on_example_chosen)
        ex_row.addWidget(self._examples, 1)
        v.addLayout(ex_row)

        # Voice row — only the "Speak responses" toggle now (the mic moved
        # into the prompt input above). Row is hidden entirely if voice
        # is unavailable.
        voice_row = self._build_voice_row()
        if voice_row is not None:
            v.addLayout(voice_row)

        # Build the overlaid mic + the auto-stop poll timer (only if voice
        # is available — keeps the prompt clean for users without it).
        self._build_overlay_mic()

        return card

    def _build_voice_row(self) -> Optional[QHBoxLayout]:
        if self._voice is None:
            return None
        can_rec = bool(getattr(self._voice, "can_record", lambda: False)())
        can_spk = bool(getattr(self._voice, "can_speak", lambda: False)())
        if not (can_rec or can_spk):
            return None

        row = QHBoxLayout()
        row.setSpacing(8)
        # The mic itself is now overlaid inside the prompt input. The voice
        # row only carries the "Speak responses" toggle below.

        # Speak-responses toggle (only meaningful if mode_ai exposes it)
        if can_spk and self._mode_ai is not None and hasattr(self._mode_ai, "set_speak_responses"):
            self._speak_btn = QPushButton("🔊  Speak responses")
            self._speak_btn.setCheckable(True)
            self._speak_btn.setProperty("variant", "ghost")
            self._speak_btn.setToolTip(
                "When ON, the planner's next-observation line is spoken via "
                "OpenAI TTS at the end of each iteration."
            )
            self._speak_btn.toggled.connect(self._on_speak_toggled)
            # Reflect the mode_ai initial state if available
            initial = bool(getattr(self._mode_ai, "speak_responses", False))
            self._speak_btn.setChecked(initial)
            row.addWidget(self._speak_btn)
        else:
            self._speak_btn = None  # type: ignore

        row.addStretch(1)
        return row

    def _build_settings_card(self, default_llm: str, default_vlm: str,
                             default_execution: str) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(12)

        cap = QLabel("PIPELINE SETTINGS")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        # Execution type segmented toggle
        exec_row = QHBoxLayout()
        exec_row.setSpacing(8)
        exec_label = QLabel("Execution")
        exec_label.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        exec_label.setMinimumWidth(76)
        exec_row.addWidget(exec_label)

        self._exec_group = QButtonGroup(self)
        self._exec_group.setExclusive(True)
        self._btn_low = QPushButton("Low-level")
        self._btn_low.setCheckable(True)
        self._btn_low.setProperty("variant", "ghost")
        self._btn_high = QPushButton("Skills")
        self._btn_high.setCheckable(True)
        self._btn_high.setProperty("variant", "ghost")
        if default_execution == "low_level":
            self._btn_low.setChecked(True)
        else:
            self._btn_high.setChecked(True)
        self._update_exec_styles()
        self._exec_group.addButton(self._btn_low)
        self._exec_group.addButton(self._btn_high)
        self._btn_low.toggled.connect(self._on_exec_toggled)
        self._btn_high.toggled.connect(self._on_exec_toggled)
        exec_row.addWidget(self._btn_low)
        exec_row.addWidget(self._btn_high)
        exec_row.addStretch(1)
        v.addLayout(exec_row)

        # LLM selector
        llm_row = QHBoxLayout()
        llm_row.setSpacing(8)
        llm_lbl = QLabel("LLM")
        llm_lbl.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        llm_lbl.setMinimumWidth(76)
        llm_row.addWidget(llm_lbl)
        self._llm_combo = QComboBox()
        for n in llm_model_names():
            self._llm_combo.addItem(n)
        self._llm_combo.setCurrentText(default_llm)
        llm_row.addWidget(self._llm_combo, 1)
        v.addLayout(llm_row)

        # VLM selector
        vlm_row = QHBoxLayout()
        vlm_row.setSpacing(8)
        vlm_lbl = QLabel("VLM")
        vlm_lbl.setStyleSheet(f"color: {Tokens.TEXT_SECONDARY}; font-size: 11px;")
        vlm_lbl.setMinimumWidth(76)
        vlm_row.addWidget(vlm_lbl)
        self._vlm_combo = QComboBox()
        for n in vlm_model_names():
            self._vlm_combo.addItem(n)
        self._vlm_combo.setCurrentText(default_vlm)
        vlm_row.addWidget(self._vlm_combo, 1)
        v.addLayout(vlm_row)

        return card

    def _build_pipeline_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        cap = QLabel("PIPELINE")
        cap.setProperty("role", "caption")
        v.addWidget(cap)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._stage_chips = {}
        for stage in PIPELINE_STAGES:
            chip = QLabel(STAGE_LABELS[stage])
            chip.setProperty("role", "stage")
            chip.setProperty("active", "false")
            chip.setProperty("done", "false")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._stage_chips[stage] = chip
            chips.addWidget(chip)
        v.addLayout(chips)
        return card

    # ---------------------------------------------------------------

    def _on_example_chosen(self, idx: int) -> None:
        if idx <= 0:
            return
        text = self._examples.currentText()
        if text and text != "— select an example —":
            self._task_input.setPlainText(text)
        self._examples.setCurrentIndex(0)

    def _on_exec_toggled(self, _checked: bool) -> None:
        self._update_exec_styles()
        et = "low_level" if self._btn_low.isChecked() else "high_level"
        self.execution_type_changed.emit(et)

    def _update_exec_styles(self) -> None:
        for btn, on in ((self._btn_low, self._btn_low.isChecked()),
                        (self._btn_high, self._btn_high.isChecked())):
            btn.setProperty("variant", "primary" if on else "ghost")
            self._restyle(btn)

    def _on_run(self) -> None:
        text = self._task_input.toPlainText().strip()
        if not text:
            return
        et = "low_level" if self._btn_low.isChecked() else "high_level"
        self.run_clicked.emit(
            text,
            self._llm_combo.currentText(),
            self._vlm_combo.currentText(),
            et,
        )

    # ---------------------------------------------------------------
    # Voice — overlaid mic button
    # ---------------------------------------------------------------

    def _build_overlay_mic(self) -> None:
        """Place a vector-drawn toggle mic button inside the prompt input's
        bottom-right corner. ALWAYS rendered — when voice isn't available
        the button is disabled with an explanatory tooltip so the operator
        can see what to install.
        """
        self._mic_btn = None
        self._mic_poll_timer = None

        viewport = self._task_input.viewport()
        self._mic_btn = _MicButton(viewport)

        can_record = (
            self._voice is not None
            and bool(getattr(self._voice, "can_record", lambda: False)())
        )
        if can_record:
            self._mic_btn.setToolTip(
                "Click to start dictating. Click again to stop, or just stop "
                "talking and the mic auto-stops after a brief silence.\n"
                "The Whisper transcript is appended to the task field."
            )
            self._mic_btn.toggled.connect(self._on_mic_toggled)
        else:
            why = "voice unavailable"
            if self._voice is None:
                why = "voice subsystem not initialised"
            else:
                status = getattr(self._voice, "status_text", lambda: "")()
                if status:
                    why = status
            self._mic_btn.setEnabled(False)
            self._mic_btn.setToolTip(
                f"Voice input disabled — {why}.\n\n"
                "To enable, restart after:\n"
                "    pip install sounddevice\n"
                "    sudo apt install libportaudio2\n"
                "and ensure api_keys.openai is set in Settings."
            )
        self._task_input.set_mic_button(self._mic_btn)
        self._task_input.reposition_mic()

        # Poll the VAD ~5 Hz while recording.
        if can_record:
            self._mic_poll_timer = QTimer(self)
            self._mic_poll_timer.setInterval(200)
            self._mic_poll_timer.timeout.connect(self._poll_auto_stop)

    def _on_mic_toggled(self, checked: bool) -> None:
        """Toggle handler — single click starts, second click stops.
        Auto-stop also routes through here by un-checking the button.
        """
        if self._voice is None:
            return
        if checked:
            ok = bool(self._voice.start_recording())
            if not ok:
                self._mic_btn.blockSignals(True)
                self._mic_btn.setChecked(False)
                self._mic_btn.blockSignals(False)
                self._set_voice_status("mic unavailable", danger=True)
                return
            self._set_voice_status("● recording — speak then pause to stop", danger=True)
            if self._mic_btn is not None:
                self._mic_btn.update()
            if self._mic_poll_timer is not None:
                self._mic_poll_timer.start()
        else:
            if self._mic_poll_timer is not None:
                self._mic_poll_timer.stop()
            if self._mic_btn is not None:
                self._mic_btn.update()
            wav = self._voice.stop_recording()
            if not wav:
                self._set_voice_status("(too short — try again)")
                return
            self._set_voice_status("transcribing…")
            try:
                asyncio.create_task(self._do_transcribe(wav))
            except RuntimeError:
                logger.warning("No running asyncio loop; transcribe skipped")
                self._set_voice_status("(no asyncio loop)")

    def _poll_auto_stop(self) -> None:
        """While recording, check the VAD heuristic and un-check the mic
        when trailing silence has lasted long enough.
        """
        if self._voice is None or self._mic_btn is None:
            return
        if not self._mic_btn.isChecked():
            if self._mic_poll_timer is not None:
                self._mic_poll_timer.stop()
            return
        if bool(getattr(self._voice, "should_auto_stop", lambda: False)()):
            # Programmatic un-check fires _on_mic_toggled(False) — same
            # path as a manual click, so transcription kicks off naturally.
            self._mic_btn.setChecked(False)

    def _set_voice_status(self, text: str, *, danger: bool = False) -> None:
        if self._voice_status is None:
            return
        colour = Tokens.DANGER if danger else Tokens.TEXT_MUTED
        self._voice_status.setStyleSheet(
            f"color: {colour}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        self._voice_status.setText(text)
        self._voice_status.setVisible(bool(text))

    async def _do_transcribe(self, wav: bytes) -> None:
        try:
            text = await self._voice.transcribe(wav)
        except Exception as exc:
            logger.exception("Transcribe failed: %s", exc)
            text = ""
        if not text:
            self._set_voice_status("(empty transcript — try again)")
            return
        # Append into the task field rather than overwrite, so the operator
        # can dictate additions over an existing prompt.
        existing = self._task_input.toPlainText().strip()
        merged = (existing + " " + text).strip() if existing else text
        self._task_input.setPlainText(merged)
        self._set_voice_status(f"✓ {len(text)} chars")

    def _on_speak_toggled(self, checked: bool) -> None:
        if self._mode_ai is None:
            return
        try:
            self._mode_ai.set_speak_responses(bool(checked))
        except Exception as exc:
            logger.exception("set_speak_responses failed: %s", exc)
            return
        if self._speak_btn is not None:
            self._speak_btn.setText(
                "🔊  Speaking on" if checked else "🔊  Speak responses"
            )
        # First time the user toggles ON, fire a short confirmation phrase so
        # they immediately hear that TTS playback works (or get an error in
        # the status strip if it doesn't).
        if checked and self._voice is not None and self._voice.can_speak():
            try:
                asyncio.create_task(self._do_test_speak())
            except RuntimeError:
                logger.warning("No running asyncio loop; test-speak skipped")

    async def _do_test_speak(self) -> None:
        ok = False
        try:
            ok = await self._voice.speak("Voice output ready.")
        except Exception as exc:
            logger.exception("Test speak failed: %s", exc)
        if not ok:
            self._set_voice_status("(TTS test failed — check audio output)", danger=True)

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        st = widget.style()
        if st is not None:
            st.unpolish(widget)
            st.polish(widget)
