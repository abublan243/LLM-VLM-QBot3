"""
AI Mode — natural-language task execution loop.

User submits a task description + selects a VLM, an LLM, and an execution
type. The mode runs an iterative perception → planning → action loop:

    sensor_fusion -> yolo -> vlm -> llm -> action -> (repeat)

On each iteration:
    1. VLMPipeline turns the latest RGB+depth+task into a structured scene
       analysis (YOLO detections, depth stats, VLM commentary).
    2. LLMPlanner turns that + robot state into a JSON action.
    3. We dispatch the action:
         action_type == "low_level"  -> publish /qbot3/cmd_vel for duration_ms
         action_type == "skill"      -> instantiate from SKILL_CLASSES, await
    4. status == "task_complete" / "blocked" exits the loop.

Hard limits:
    * 30 iterations per task (safety against runaway planners)
    * 5 minute wall-clock cap per task

Pipeline stage signals are emitted so the GUI status bar can light up the
current stage. Task records are appended to SharedState.task_history at
start and completion.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.shared_state import SharedState, TaskRecord
from skills import (
    SKILL_CLASSES,
    BaseSkill,
    planner_visible_skills,
    skill_metadata,
)

logger = logging.getLogger(__name__)


# Stage names emitted via pipeline_stage. The GUI lights up each in turn.
STAGES = ("sensor_fusion", "yolo", "vlm", "llm", "action")


class ModeAI(QObject):
    """AI control loop coordinator. Owns the VLM/LLM pipeline references."""

    pipeline_stage = pyqtSignal(str)                # one of STAGES
    iteration_started = pyqtSignal(int)             # iteration index (1-based)
    task_started = pyqtSignal(str)                  # task description
    task_completed = pyqtSignal(bool, str)          # success, message
    report_ready = pyqtSignal(str)                  # path to generated PDF (or .tex)
    activated = pyqtSignal()
    deactivated = pyqtSignal()

    DEFAULT_MAX_ITERATIONS = 30
    DEFAULT_WALL_CLOCK_S = 300.0

    def __init__(
        self,
        state: SharedState,
        ros: Any,
        vlm_pipeline: Any,
        llm_planner: Any,
        *,
        skills_config: Optional[Dict[str, Any]] = None,
        performance_monitor: Optional[Any] = None,
        slam_manager: Optional[Any] = None,
        llm_model_name: str = "gpt-4o-mini",
        vlm_model_name: str = "gpt-4o",
        execution_type: str = "high_level",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        wall_clock_s: float = DEFAULT_WALL_CLOCK_S,
        voice_io: Optional[Any] = None,
        speak_responses: bool = False,
        speak_section: str = "next_observation",
        max_speak_chars: int = 240,
        generate_reports: bool = True,
        vlm_grounding: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros
        self.vlm = vlm_pipeline
        self.llm = llm_planner
        self.skills_config = skills_config or {}
        self._perf = performance_monitor
        self._slam = slam_manager
        # Round 20: visual ROI grounding cascade injected by main.py.
        # `vlm_reach` reads this at instantiation time. None == feature off.
        self._vlm_grounding = vlm_grounding
        self.generate_reports = bool(generate_reports)

        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        self.execution_type = execution_type     # "high_level" | "low_level"
        self.max_iterations = max_iterations
        self.wall_clock_s = wall_clock_s

        # Voice (optional). When `speak_responses` is True the loop hands the
        # chosen LLM section to voice_io.speak() per iteration. Settings dialog
        # / control_panel can flip this at runtime via set_speak_responses().
        self.voice = voice_io
        self.speak_responses = bool(speak_responses)
        self.speak_section = speak_section
        self.max_speak_chars = max(40, int(max_speak_chars))

        self._task: Optional[asyncio.Task] = None
        self._active_skill: Optional[BaseSkill] = None
        self._cancel: bool = False
        self._task_record: Optional[TaskRecord] = None
        self._mission: Optional[Any] = None        # MissionRecorder per task

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def activate(self) -> None:
        self.state.set_active_mode("ai")
        self.activated.emit()

    def deactivate(self) -> None:
        self.cancel_task()
        self.deactivated.emit()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    def set_models(self, llm_name: str, vlm_name: str) -> None:
        self.llm_model_name = llm_name
        self.vlm_model_name = vlm_name

    def set_execution_type(self, execution_type: str) -> None:
        if execution_type not in ("low_level", "high_level"):
            logger.warning("Unknown execution_type: %s", execution_type)
            return
        self.execution_type = execution_type

    def set_api_keys(self, openai_key: str, google_key: str) -> None:
        self.vlm.set_api_keys(openai_key, google_key)
        self.llm.set_api_keys(openai_key, google_key)

    def set_yolo_confidence(self, confidence: float) -> None:
        self.vlm.set_confidence(confidence)

    def set_speak_responses(self, enabled: bool) -> None:
        self.speak_responses = bool(enabled)
        if not enabled and self.voice is not None:
            try:
                self.voice.stop_speaking()
            except Exception:
                pass

    def _maybe_speak(self, llm_output: Any) -> None:
        """Pick the configured field off the LLM output and hand to TTS."""
        if not self.speak_responses or self.voice is None:
            return
        if not getattr(self.voice, "can_speak", lambda: False)():
            return
        if self.speak_section == "next_observation":
            text = getattr(llm_output, "next_observation", "") or ""
        elif self.speak_section == "reasoning":
            text = getattr(llm_output, "reasoning", "") or ""
        elif self.speak_section == "status":
            text = getattr(llm_output, "status", "") or ""
        else:
            text = ""
        text = text.strip()
        if not text:
            return
        if len(text) > self.max_speak_chars:
            text = text[: self.max_speak_chars - 1].rstrip() + "…"
        try:
            asyncio.create_task(self.voice.speak(text))
        except Exception:
            logger.exception("voice.speak schedule failed")

    # ---------------------------------------------------------------
    # Task entry / cancel
    # ---------------------------------------------------------------

    def submit_task(self, task_description: str) -> bool:
        """Spawn a run_task coroutine on the current event loop."""
        if self.is_running:
            logger.warning("Task already running — ignoring submit")
            return False
        # Refuse to start while the Pi gyro is still calibrating — the Pi
        # silently drops cmd_vel during that window, so a task would burn
        # tokens with no motion. Better to fail fast with a clear message.
        with self.state.lock:
            calibrated = self.state.imu_calibrated
        if not calibrated:
            logger.warning("AI task refused — gyro still calibrating")
            self.task_completed.emit(False, "gyroscope calibration in progress — wait for the banner")
            return False
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._task = loop.create_task(self.run_task(task_description))
        return True

    def cancel_task(self) -> None:
        """Abort the active skill (if any) and signal the loop to exit."""
        self._cancel = True
        if self._active_skill is not None:
            try:
                self._active_skill.abort()
            except Exception:
                pass
        try:
            self.ros.publish_emergency_stop()
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------

    async def run_task(self, task_description: str) -> None:
        """Run the perception/plan/act loop until task_complete, blocked, or cancelled."""
        task_description = (task_description or "").strip()
        if not task_description:
            self.task_completed.emit(False, "empty task")
            return

        self._cancel = False
        self._task_record = TaskRecord(
            description=task_description,
            started_ts=time.monotonic(),
            status="running",
        )
        self.state.append_task(self._task_record)
        with self.state.lock:
            self.state.active_task_description = task_description
            has_base = "base" in self.state.named_waypoints
        # If the operator never saved a 'base' waypoint, treat the mission
        # start pose as base. Otherwise any task ending in "return to base"
        # (the APPROACH+RETURN and SEQUENCE+RETURN tiers) would fail with
        # "no 'base' waypoint saved yet" and the robot could never complete.
        if not has_base:
            self.state.save_named_waypoint("base")
            logger.info("No 'base' waypoint set — captured task start pose as base")
        self.task_started.emit(task_description)
        logger.info("AI task started: %s", task_description)

        # Spin up a fresh mission recorder. Failures here are not fatal —
        # the task runs even if the recorder can't be constructed.
        self._mission = None
        if self.generate_reports:
            try:
                from reports import MissionRecorder
                self._mission = MissionRecorder(
                    self.state,
                    slam_manager=self._slam,
                    performance_monitor=self._perf,
                    ros_bridge=self.ros,
                    vlm_pipeline=self.vlm,
                )
                self._mission.start(
                    task_description,
                    llm_model=self.llm_model_name,
                    vlm_model=self.vlm_model_name,
                    execution_type=self.execution_type,
                )
            except Exception as exc:
                logger.warning("MissionRecorder unavailable: %s", exc)
                self._mission = None

        deadline = time.monotonic() + self.wall_clock_s
        success = False
        message = ""
        # Only the skills the LLM is allowed to pick — hides legacy /
        # unreliable ones (wall_follower, line_follower) without removing
        # them from the codebase. They remain runnable from block-programming.
        _visible = planner_visible_skills()
        allowed_skills: List[str] = list(_visible.keys())
        _all_meta = skill_metadata()
        skill_descs = {n: m for n, m in _all_meta.items() if n in _visible}

        try:
            for iteration in range(1, self.max_iterations + 1):
                if self._cancel:
                    message = "cancelled by operator"
                    break
                if time.monotonic() > deadline:
                    message = f"wall-clock timeout ({self.wall_clock_s:.0f}s)"
                    break

                self.iteration_started.emit(iteration)
                if self._mission is not None:
                    try:
                        self._mission.begin_iteration(iteration)
                    except Exception:
                        logger.exception("recorder.begin_iteration failed")

                # Stage 1: sensor fusion (brief — frame snapshots happen inside vlm.run_async)
                self.pipeline_stage.emit("sensor_fusion")
                await asyncio.sleep(0)
                if not self._frames_ready():
                    await asyncio.sleep(0.2)
                    if not self._frames_ready():
                        message = "no camera frames yet — is the bridge connected?"
                        break

                # Stage 2 + 3: YOLO + VLM (both happen inside the pipeline)
                self.pipeline_stage.emit("yolo")
                await asyncio.sleep(0)
                self.pipeline_stage.emit("vlm")
                try:
                    pipeline_result = await self.vlm.run_async(
                        task_description, self.vlm_model_name,
                    )
                except Exception as exc:
                    logger.exception("VLM stage failed: %s", exc)
                    message = f"VLM stage failed: {type(exc).__name__}: {exc}"
                    break

                vlm_text = pipeline_result["vlm_output"].raw_text
                if self._mission is not None:
                    try:
                        self._mission.attach_vlm_output(
                            iteration, pipeline_result["vlm_output"],
                        )
                        # Capture annotated frame from the pipeline if present
                        annotated = pipeline_result.get("annotated_frame")
                        if annotated is not None:
                            self._mission.capture(
                                f"yolo_iter{iteration}", annotated,
                            )
                    except Exception:
                        logger.exception("recorder VLM attach failed")

                # Stage 4: LLM planning
                self.pipeline_stage.emit("llm")
                try:
                    llm_output = await self.llm.plan_async(
                        task_description,
                        vlm_text,
                        self.llm_model_name,
                        execution_type=self.execution_type,
                        allowed_skills=allowed_skills,
                        skill_descriptions=skill_descs,
                    )
                except Exception as exc:
                    logger.exception("LLM stage failed: %s", exc)
                    message = f"LLM stage failed: {type(exc).__name__}: {exc}"
                    break

                if self._mission is not None:
                    try:
                        self._mission.attach_llm_output(iteration, llm_output)
                        # Per-iteration SLAM snapshot (covers the planner's
                        # view of the map at decision time)
                        if self._slam is not None:
                            self._mission.capture(
                                f"slam_iter{iteration}",
                                self._slam.get_map_image(),
                            )
                    except Exception:
                        logger.exception("recorder LLM attach failed")

                # Voice: speak the planner's chosen section before we act —
                # gives the operator narrative even on long-running iterations.
                # Fire-and-forget; never block the control loop on TTS latency.
                self._maybe_speak(llm_output)

                # Terminal statuses
                if llm_output.status == "task_complete":
                    success = True
                    message = "task complete"
                    break
                if llm_output.status == "blocked":
                    message = f"blocked: {llm_output.reasoning[:200]}"
                    break

                # Stage 5: action dispatch
                self.pipeline_stage.emit("action")
                if llm_output.action_type == "low_level":
                    await self._execute_low_level(llm_output.low_level_command or {},
                                                  decision_ts=time.monotonic())
                elif llm_output.action_type == "skill":
                    if self.execution_type == "low_level":
                        # Skill suggested but operator restricted us to low-level — stop
                        message = "planner asked for skill in low-level mode; halting"
                        break
                    skill_ok = await self._execute_skill(llm_output.skill_command or {})
                    if not skill_ok and self._cancel:
                        message = "cancelled during skill"
                        break
                else:
                    logger.warning("Unknown action_type: %s", llm_output.action_type)
                    # Treat as no-op and re-plan

                if self._mission is not None:
                    try:
                        self._mission.end_iteration(iteration)
                    except Exception:
                        logger.exception("recorder.end_iteration failed")

            else:
                # for/else — exhausted iterations without break
                message = f"reached iteration cap ({self.max_iterations})"

        finally:
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass
            if self._task_record is not None:
                self._task_record.finished_ts = time.monotonic()
                self._task_record.status = "success" if success else "failed"
                self._task_record.notes = message
            with self.state.lock:
                self.state.active_task_description = ""
            if self._perf is not None:
                try:
                    self._perf.record_task_outcome(success)
                except Exception:
                    pass

            # Finalise and export the mission report. Runs synchronously
            # inside a thread so pdflatex (1–3 s) never blocks the qasync loop.
            if self._mission is not None:
                try:
                    self._mission.end(success, message)
                    report_path = await asyncio.to_thread(self._mission.export)
                    if report_path is not None:
                        logger.info("Mission report at %s", report_path)
                        try:
                            self.report_ready.emit(str(report_path))
                        except Exception:
                            pass
                except Exception:
                    logger.exception("Mission report export failed")
                finally:
                    self._mission = None

        logger.info("AI task finished: success=%s message=%s", success, message)
        self.task_completed.emit(success, message)

    # ---------------------------------------------------------------
    # Action dispatchers
    # ---------------------------------------------------------------

    async def _execute_low_level(self, cmd: Dict[str, Any], *,
                                 decision_ts: float) -> None:
        """Publish cmd_vel at 10 Hz for duration_ms, then stop."""
        linear_x = float(cmd.get("linear_x", 0.0) or 0.0)
        angular_z = float(cmd.get("angular_z", 0.0) or 0.0)
        duration_ms = int(cmd.get("duration_ms", 500) or 500)
        duration_s = max(0.05, min(5.0, duration_ms / 1000.0))

        # Latency: from LLM decision → first publish
        publish_ts = time.monotonic()
        if self._perf is not None:
            try:
                self._perf.record_motor_cmd((publish_ts - decision_ts) * 1000.0)
            except Exception:
                pass

        end = publish_ts + duration_s
        try:
            while time.monotonic() < end and not self._cancel:
                self.ros.publish_cmd_vel(linear_x, angular_z)
                with self.state.lock:
                    bumper = self.state.bumpers.any_active()
                    cliff = self.state.cliff.any_active()
                    wheel_drop = self.state.wheel_drop.any_active()
                if bumper or cliff or wheel_drop:
                    break
                await asyncio.sleep(0.1)
        finally:
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass

    async def _execute_skill(self, skill_cmd: Dict[str, Any]) -> bool:
        """Instantiate the requested skill and await completion. Returns success."""
        name = str(skill_cmd.get("skill_name", "")).strip()
        params = skill_cmd.get("parameters") or {}
        if name not in SKILL_CLASSES:
            logger.warning("Planner asked for unknown skill: %s", name)
            self.state.append_action({
                "action_type": "skill_result",
                "skill_name": name,
                "params": params if isinstance(params, dict) else {},
                "success": False,
                "message": f"unknown skill '{name}'",
            })
            return False

        # `cls` is Any so subclass-specific kwargs (read_gauge's
        # vlm_pipeline, vlm_reach's vlm_grounding) don't trip the type
        # checker against BaseSkill.__init__'s narrower signature.
        cls: Any = SKILL_CLASSES[name]
        skill: BaseSkill
        if name == "read_gauge":
            skill = cls(
                self.state, self.ros,
                vlm_pipeline=self.vlm,
                vlm_model_name=self.vlm_model_name,
                skills_config=self.skills_config,
            )
        elif name == "vlm_reach":
            skill = cls(
                self.state, self.ros,
                skills_config=self.skills_config,
                vlm_grounding=self._vlm_grounding,
            )
        elif name == "speak_text":
            skill = cls(
                self.state, self.ros,
                voice_io=self.voice,
                skills_config=self.skills_config,
            )
        else:
            skill = cls(self.state, self.ros, skills_config=self.skills_config)

        self._active_skill = skill
        with self.state.lock:
            self.state.active_skill = name
        runtime_params = params if isinstance(params, dict) else {}
        try:
            result = await skill.run(runtime_params)
            # Surface the outcome to the next planner iteration. Without this
            # the LLM cannot see WHY a skill failed (e.g. missing target_class)
            # and re-picks the same broken action forever.
            self.state.append_action({
                "action_type": "skill_result",
                "skill_name": name,
                "params": runtime_params,
                "success": bool(result.success),
                "message": str(result.message)[:200],
            })
            return bool(result.success)
        except Exception as exc:
            logger.exception("Skill '%s' raised: %s", name, exc)
            self.state.append_action({
                "action_type": "skill_result",
                "skill_name": name,
                "params": runtime_params,
                "success": False,
                "message": f"raised {type(exc).__name__}: {exc}"[:200],
            })
            return False
        finally:
            with self.state.lock:
                self.state.active_skill = None
                self.state.active_skill_progress = 0.0
            self._active_skill = None

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _frames_ready(self) -> bool:
        with self.state.lock:
            return self.state.rgb_frame is not None
