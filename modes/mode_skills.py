"""
Skills Mode — runs a single user-selected BaseSkill at a time.

The mode wraps the asyncio Task lifecycle so the GUI can fire-and-forget
("Run wall_follower"), then call pause/resume/abort without dealing with
asyncio internals. Progress + status from the active skill are reflected
into SharedState every 200 ms so the GUI status bar / cards can read them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.shared_state import SharedState, SkillRecord
from skills import SKILL_CLASSES, BaseSkill, skill_metadata

logger = logging.getLogger(__name__)


class SkillsMode(QObject):
    """Coordinator that owns at most one running skill instance + asyncio.Task."""

    skill_started = pyqtSignal(str)               # skill name
    skill_progress = pyqtSignal(float, str)       # 0..1, status string
    skill_finished = pyqtSignal(str, bool, str)   # name, success, message
    activated = pyqtSignal()
    deactivated = pyqtSignal()

    def __init__(
        self,
        state: SharedState,
        ros: Any,
        *,
        skills_config: Optional[Dict[str, Any]] = None,
        vlm_pipeline: Optional[Any] = None,
        vlm_model_name: str = "gpt-4o",
        vlm_grounding: Optional[Any] = None,
        voice_io: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros
        self.skills_config = skills_config or {}
        self._vlm_pipeline = vlm_pipeline
        self._vlm_model_name = vlm_model_name
        # Round 20: VLM grounding cascade for the new vlm_reach skill.
        self._vlm_grounding = vlm_grounding
        self._voice_io = voice_io

        self._active_skill: Optional[BaseSkill] = None
        self._active_task: Optional[asyncio.Task] = None
        self._active_started_ts: float = 0.0
        self._active_params: Dict[str, Any] = {}

        # Reflect skill progress -> shared state -> GUI
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_progress)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def activate(self) -> None:
        self.state.set_active_mode("skills")
        self.activated.emit()

    def deactivate(self) -> None:
        # Always abort whatever is running on mode switch
        if self._active_task is not None:
            self.abort_active()
        self.deactivated.emit()

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    def set_vlm(self, pipeline: Any, model_name: str = "gpt-4o") -> None:
        self._vlm_pipeline = pipeline
        self._vlm_model_name = model_name

    # ---------------------------------------------------------------
    # Skill control
    # ---------------------------------------------------------------

    def list_skills(self) -> Dict[str, Dict[str, str]]:
        return skill_metadata()

    @property
    def active_name(self) -> Optional[str]:
        return self._active_skill.name if self._active_skill else None

    def run_skill(self, name: str, params: Optional[Dict[str, Any]] = None
                  ) -> bool:
        """Spawn a skill on the running asyncio loop. Returns True on accept."""
        if self._active_task is not None and not self._active_task.done():
            logger.warning("Cannot start '%s': '%s' still running",
                           name, self.active_name)
            return False
        if name not in SKILL_CLASSES:
            logger.error("Unknown skill: %s", name)
            return False
        # Refuse during gyro calibration — Pi will silently drop motion.
        with self.state.lock:
            calibrated = self.state.imu_calibrated
        if not calibrated:
            logger.warning("Skill '%s' refused — gyro still calibrating", name)
            self.skill_finished.emit(name, False,
                                     "gyroscope calibration in progress — wait for the banner")
            return False

        skill = self._instantiate(name)
        self._active_skill = skill
        self._active_params = params or {}
        self._active_started_ts = time.monotonic()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._active_task = loop.create_task(self._run_and_finalize(skill, params or {}))
        with self.state.lock:
            self.state.active_skill = name
            self.state.active_skill_progress = 0.0
        self._poll_timer.start()
        self.skill_started.emit(name)
        return True

    def pause_active(self) -> None:
        if self._active_skill is not None:
            self._active_skill.pause()

    def resume_active(self) -> None:
        if self._active_skill is not None:
            self._active_skill.resume()

    def abort_active(self) -> None:
        if self._active_skill is not None:
            self._active_skill.abort()
        # Don't cancel the task — let it return cleanly via the abort flag

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _instantiate(self, name: str) -> BaseSkill:
        # `cls` is intentionally typed Any: per-skill subclasses accept
        # different injected dependencies (read_gauge needs vlm_pipeline +
        # vlm_model_name; vlm_reach needs vlm_grounding) — Pyright would
        # otherwise reject the kwarg names against BaseSkill.__init__.
        cls: Any = SKILL_CLASSES[name]
        if name == "read_gauge":
            return cls(
                self.state, self.ros,
                vlm_pipeline=self._vlm_pipeline,
                vlm_model_name=self._vlm_model_name,
                skills_config=self.skills_config,
            )
        if name == "vlm_reach":
            return cls(
                self.state, self.ros,
                skills_config=self.skills_config,
                vlm_grounding=self._vlm_grounding,
            )
        if name == "speak_text":
            return cls(
                self.state, self.ros,
                voice_io=self._voice_io,
                skills_config=self.skills_config,
            )
        return cls(self.state, self.ros, skills_config=self.skills_config)

    async def _run_and_finalize(self, skill: BaseSkill,
                                params: Dict[str, Any]) -> None:
        success = False
        message = ""
        try:
            result = await skill.run(params)
            success = bool(result.success)
            message = result.message or ""
        except asyncio.CancelledError:
            success = False
            message = "cancelled"
            raise
        except Exception as exc:
            logger.exception("Skill '%s' top-level crash: %s", skill.name, exc)
            success = False
            message = f"{type(exc).__name__}: {exc}"
        finally:
            self.state.append_skill(SkillRecord(
                name=skill.name,
                started_ts=self._active_started_ts,
                finished_ts=time.monotonic(),
                success=success,
                parameters=dict(params),
            ))
            with self.state.lock:
                self.state.active_skill = None
                self.state.active_skill_progress = 0.0
            self._poll_timer.stop()
            self._active_skill = None
            self._active_task = None
            self.skill_finished.emit(skill.name, success, message)

    def _poll_progress(self) -> None:
        if self._active_skill is None:
            return
        progress = self._active_skill.get_progress()
        status = self._active_skill.get_status_string()
        with self.state.lock:
            self.state.active_skill_progress = progress
        self.skill_progress.emit(progress, status)
