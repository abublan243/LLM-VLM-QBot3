"""
Search Object — rotate the robot in steps; at each step check shared_state's
detection list (populated by the VLM/YOLO pipeline) for a target class.

The pipeline is expected to be running in parallel (mode_ai keeps it ticking,
or mode_skills can opt in). If no fresh detections arrive, the skill polls
SharedState and times out after `max_full_rotations`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from skills.base_skill import BaseSkill, SkillResult


class SearchObjectSkill(BaseSkill):
    name = "search_object"
    description = "Rotate in place looking for a target YOLO class."
    icon = "search_object"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_class", "")).lower()
        if not target:
            return SkillResult(success=False, message="no target_class given")

        rotation_speed = float(params.get("rotation_speed_radps", 0.6))
        step_deg = float(params.get("step_angle_deg", 30))
        confidence = float(params.get("confidence_threshold", 0.5))
        max_rotations = int(params.get("max_full_rotations", 2))

        total_to_cover = math.radians(360.0 * max_rotations)
        rotated = 0.0
        last_yaw: Optional[float] = None
        self._set(progress=0.05, status=f"searching for '{target}'")

        while not self._aborted and rotated < total_to_cover:
            await self._tick(0.05)

            # Step rotate by step_deg using continuous /cmd_vel
            step_rad = math.radians(step_deg)
            step_duration_s = max(0.5, abs(step_rad) / max(0.1, rotation_speed))

            # Drive at rotation_speed for step_duration_s
            t_start = time.monotonic()
            while time.monotonic() - t_start < step_duration_s:
                await self._drive(0.0, rotation_speed)
                await self._tick(0.05)
                if self._target_visible(target, confidence):
                    await self._stop_drive()
                    return SkillResult(
                        success=True,
                        message=f"found '{target}'",
                        payload={"target_class": target},
                    )

            await self._drive(0.0, 0.0)
            await self._tick(0.4)        # let YOLO catch up at the new heading

            if self._target_visible(target, confidence):
                return SkillResult(
                    success=True,
                    message=f"found '{target}'",
                    payload={"target_class": target},
                )

            # Track how far we've rotated using odom yaw (more accurate than dt × rate)
            with self.state.lock:
                yaw = self.state.odom.yaw_rad
            if last_yaw is not None:
                rotated += abs(self._wrap_pi(yaw - last_yaw))
            last_yaw = yaw

            self._set(progress=min(0.95, rotated / total_to_cover))

        return SkillResult(success=False, message=f"target '{target}' not seen")

    def _target_visible(self, target: str, confidence: float) -> bool:
        with self.state.lock:
            for det in self.state.detected_objects:
                if det.class_name.lower() == target and det.confidence >= confidence:
                    return True
        return False
