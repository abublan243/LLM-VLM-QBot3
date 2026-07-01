"""
BaseSkill — abstract async skill class. All concrete skills inherit from this.

A skill is a long-running coroutine that:
    * reads sensor data from SharedState
    * publishes velocity / precise-motion commands via the ROS bridge
    * reports progress (0–1) and a status string the GUI can render
    * can be paused, resumed, or aborted at any time
    * returns a SkillResult when it finishes (success / failure / aborted)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.shared_state import SharedState

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Outcome of one skill run."""
    success: bool
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


class SkillAborted(Exception):
    """Raised internally when the operator aborts the skill mid-run."""


class BaseSkill:
    """Subclasses must override `name`, `description`, and `_execute()`."""

    name: str = "base"
    description: str = "Override me"
    icon: str = "base"            # GUI resolves to assets/icons/<icon>.png

    def __init__(
        self,
        state: SharedState,
        ros: Any,
        *,
        skills_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = state
        self.ros = ros
        # Keep the FULL config so we can pass it to delegate skills (e.g. when
        # read_gauge spawns approach_object). Without this each delegate ran
        # with default-zero params and ignored skills_config.yaml entirely.
        self._skills_config: Dict[str, Any] = dict(skills_config or {})
        self._defaults: Dict[str, Any] = self._skills_config.get(self.name, {})

        self._progress: float = 0.0
        self._status: str = "idle"
        self._aborted: bool = False
        self._abort_reason: str = ""
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()  # not paused by default
        self._task: Optional[asyncio.Task] = None
        # Round 20: bumper-hit handler is automatic for every skill.
        # `_bumper_handled` is a one-shot latch so we don't re-handle
        # the same physical contact on every 50 ms tick — it clears
        # when the bumper state returns to inactive.
        self._bumper_handled: bool = False

    # ---------------------------------------------------------------
    # Lifecycle — called by mode_skills.py and the GUI
    # ---------------------------------------------------------------

    async def run(self, params: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Top-level entry point — wraps `_execute` with cleanup + safety stop."""
        merged = dict(self._defaults)
        if params:
            merged.update(params)

        self._aborted = False
        self._pause_event.set()
        self._progress = 0.0
        self._status = "starting"
        logger.info("Skill '%s' starting with params: %s", self.name, merged)

        try:
            result = await self._execute(merged)
        except SkillAborted:
            # `_abort_reason` is populated by the bumper handler (and any
            # future automatic abort source). If empty, this was an
            # operator-driven abort.
            reason = self._abort_reason or "aborted by operator"
            result = SkillResult(success=False, message=reason)
        except asyncio.CancelledError:
            result = SkillResult(success=False, message="cancelled")
            raise
        except Exception as exc:
            logger.exception("Skill '%s' crashed: %s", self.name, exc)
            result = SkillResult(success=False, message=f"{type(exc).__name__}: {exc}")
        finally:
            # Always leave the robot stopped at the end of a skill
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass
            self._status = "done" if (result and result.success) else "failed"

        self._progress = 1.0
        return result

    def pause(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._status = "paused"
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass

    def resume(self) -> None:
        if not self._pause_event.is_set():
            self._pause_event.set()
            self._status = "running"

    def abort(self) -> None:
        self._aborted = True
        self._pause_event.set()        # un-stick any wait
        try:
            self.ros.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # GUI accessors
    # ---------------------------------------------------------------

    def get_progress(self) -> float:
        return max(0.0, min(1.0, self._progress))

    def get_status_string(self) -> str:
        return self._status

    # ---------------------------------------------------------------
    # Subclass override
    # ---------------------------------------------------------------

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Helpers shared by all skills
    # ---------------------------------------------------------------

    async def _tick(self, dt: float = 0.05) -> None:
        """Sleep for `dt`; respect pause and abort. Raises SkillAborted on abort.

        Also runs the auto bumper-hit handler — every skill inherits the
        same reactive: stamp the obstacle on the SLAM grid, reverse a
        short distance, abort with a "bumped" message so the AI-mode
        planner sees the failure and replans the rest of the task.
        """
        if self._aborted:
            raise SkillAborted()
        await self._maybe_handle_bumper()
        if self._aborted:
            raise SkillAborted()
        if not self._pause_event.is_set():
            await self._pause_event.wait()
        await asyncio.sleep(dt)
        if self._aborted:
            raise SkillAborted()

    async def _maybe_handle_bumper(self) -> None:
        """If the bumper just fired, mark the wall on the SLAM grid,
        reverse a short distance, and abort. One-shot latch prevents
        re-triggering at every tick while the bumper remains active.
        """
        with self.state.lock:
            active = self.state.bumpers.active()    # [L, C, R]
        if not any(active):
            # Bumper has released — re-arm the latch for the next contact.
            self._bumper_handled = False
            return
        if self._bumper_handled:
            return
        self._bumper_handled = True

        # Map the channel that fired to a world (x, y) position so we
        # can stamp the wall on the SLAM grid.
        slam = getattr(self.state, "slam_manager", None)
        with self.state.lock:
            rx = self.state.odom.x
            ry = self.state.odom.y
            ryaw = self.state.odom.yaw_rad
        # Bumper sits ~16 cm forward of base_link. Add a lateral offset
        # of ±5 cm for the L/R channel so the stamp lands where the
        # contact actually was.
        forward_m = 0.16
        side_m = 0.0
        if active[0] and not active[2]:
            side_m = +0.05      # left
        elif active[2] and not active[0]:
            side_m = -0.05      # right
        # base_link → world
        c = math.cos(ryaw); s = math.sin(ryaw)
        hit_x = rx + c * forward_m - s * side_m
        hit_y = ry + s * forward_m + c * side_m

        if slam is not None and hasattr(slam, "stamp_obstacle"):
            try:
                slam.stamp_obstacle(hit_x, hit_y, radius_m=0.18, weight=4.0)
            except Exception as exc:
                logger.warning("Bumper SLAM stamp failed: %s", exc)
        else:
            logger.warning("Bumper hit but no slam_manager attached")

        # Reverse a short distance — synchronous burst, no precise_cmd
        # round-trip needed (we want to GET OFF the wall fast).
        try:
            self.ros.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass
        reverse_speed = -0.10
        reverse_duration_s = 0.8
        try:
            t_end = time.monotonic() + reverse_duration_s
            while time.monotonic() < t_end:
                self.ros.publish_cmd_vel(reverse_speed, 0.0)
                await asyncio.sleep(0.05)
        finally:
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass
        # Tell SharedState to release the latches now that we've reversed.
        with self.state.lock:
            self.state.bumpers.clear_all()

        sides = []
        if active[0]:
            sides.append("L")
        if active[1]:
            sides.append("C")
        if active[2]:
            sides.append("R")
        self._abort_reason = (
            f"bumped ({'+'.join(sides)}); marked obstacle at "
            f"({hit_x:+.2f},{hit_y:+.2f}) and reversed {abs(reverse_speed) * reverse_duration_s:.2f}m"
        )
        logger.info("Bumper handler: %s", self._abort_reason)
        self._aborted = True
        self._status = "bumped"

    async def _drive(self, linear_x: float, angular_z: float) -> None:
        """Publish one /qbot3/cmd_vel sample (velocity is held by the Pi until next message)."""
        if self._aborted:
            raise SkillAborted()
        self.ros.publish_cmd_vel(float(linear_x), float(angular_z))

    async def _stop_drive(self) -> None:
        try:
            self.ros.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass

    async def _run_precise_command(
        self,
        *,
        distance_m: float = 0.0,
        angle_rad: float = 0.0,
        timeout_s: float = 30.0,
    ) -> bool:
        """Send a goal on /qbot3/precise_cmd and wait for the controller's result.

        Returns True on success, False on timeout/failure/abort.
        Behaviour matches qbotpi/motion_controller.py: status transitions
        idle -> moving|turning -> idle and a Bool published on /motion/result.
        """
        with self.state.lock:
            self.state.motion_feedback.last_result = None
            self.state.motion_feedback.status = "idle"

        self.ros.publish_precise_cmd(distance_m=distance_m, angle_rad=angle_rad)

        # Wait for the controller to leave idle ("moving"/"turning" both signal acceptance)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            await self._tick(0.05)
            if self.state.motion_feedback.status in ("moving", "turning", "emergency_stop"):
                break

        # Wait for the controller to return to idle (or emergency_stop) and publish a result
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await self._tick(0.1)
            with self.state.lock:
                status = self.state.motion_feedback.status
                result = self.state.motion_feedback.last_result
            if status in ("idle", "emergency_stop") and result is not None:
                return bool(result) and status != "emergency_stop"

        # Timeout — issue an emergency stop and report failure
        try:
            self.ros.publish_emergency_stop()
        except Exception:
            pass
        return False

    # ---- Common geometry helpers ----

    @staticmethod
    def _wrap_pi(angle_rad: float) -> float:
        a = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
        return a

    def _heading_error_to(self, target_x: float, target_y: float) -> float:
        with self.state.lock:
            o = self.state.odom
            x, y, yaw = o.x, o.y, o.yaw_rad
        desired = math.atan2(target_y - y, target_x - x)
        return self._wrap_pi(desired - yaw)

    def _distance_to(self, target_x: float, target_y: float) -> float:
        with self.state.lock:
            o = self.state.odom
            x, y = o.x, o.y
        dx = target_x - x
        dy = target_y - y
        return math.hypot(dx, dy)

    def _set(self, *, progress: Optional[float] = None,
             status: Optional[str] = None) -> None:
        if progress is not None:
            self._progress = max(0.0, min(1.0, progress))
        if status is not None:
            self._status = status
