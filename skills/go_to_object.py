"""
Go-To-Object — drive to a remembered object by name.

Looks the target up in the persistent ``ObjectMemory`` (the RAG layer
populated by the continuous YOLO loop), computes an approach point
``stop_distance_m`` short of the object's recorded position along the
line from the current robot pose, and delegates the navigation to
``GoToPositionSkill``. Optionally turns to face the object at the end.

Use this when the operator says "go to the chair" and the robot has
already seen one in this session — much faster than scanning the room
again with ``search_object`` + ``approach_object``.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from skills.base_skill import BaseSkill, SkillResult
from skills.go_to_position import GoToPositionSkill


class GoToObjectSkill(BaseSkill):
    name = "go_to_object"
    description = "Navigate to a previously-seen object by class name (RAG)."
    icon = "go_to_object"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_name", "")).strip().lower()
        if not target:
            return SkillResult(success=False, message="no target_name given")

        memory = getattr(self.state, "object_memory", None)
        if memory is None:
            return SkillResult(
                success=False,
                message="object_memory not initialised — RAG layer disabled",
            )

        stop_distance = float(params.get(
            "stop_distance_m",
            self._defaults.get("stop_distance_m", 0.5),
        ))
        instance = int(params.get("instance_index", 0))
        face_target = bool(params.get(
            "face_target", self._defaults.get("face_target", True),
        ))
        tolerance = float(params.get(
            "tolerance_m", self._defaults.get("tolerance_m", 0.10),
        ))

        entry = memory.find(target, instance=instance)
        if entry is None:
            available = ", ".join(memory.class_names()) or "(empty)"
            return SkillResult(
                success=False,
                message=(f"no remembered '{target}' (known classes: {available})"),
            )

        # ---- Approach point: stop_distance short of the recorded object
        # along the line from the robot's CURRENT pose. We compute it once,
        # at start; if the robot's odom drifts during the drive, the final
        # bearing-only correction below still lines up the heading.
        with self.state.lock:
            rx = self.state.odom.x
            ry = self.state.odom.y

        dx = entry.x - rx
        dy = entry.y - ry
        dist = math.hypot(dx, dy)
        self._set(
            progress=0.05,
            status=f"target '{target}' @ ({entry.x:+.2f}, {entry.y:+.2f}) "
                   f"d={dist:.2f}m",
        )

        if dist <= stop_distance + tolerance:
            approach_x, approach_y = rx, ry
            already_close = True
        else:
            scale = (dist - stop_distance) / dist
            approach_x = rx + dx * scale
            approach_y = ry + dy * scale
            already_close = False

        # ---- Delegate to go_to_position ----
        if not already_close:
            delegate = GoToPositionSkill(
                self.state, self.ros,
                skills_config=self._skills_config,
            )
            delegate._pause_event = self._pause_event
            delegate._aborted = False
            leg = await delegate.run({
                "x": approach_x,
                "y": approach_y,
                "tolerance_m": tolerance,
                "use_precise_cmd": True,
            })
            if not leg.success:
                return SkillResult(
                    success=False,
                    message=f"go_to_position failed: {leg.message}",
                )

        # ---- Final heading: face the object ----
        if face_target:
            self._set(progress=0.9, status=f"facing '{target}'")
            with self.state.lock:
                rx2 = self.state.odom.x
                ry2 = self.state.odom.y
                ryaw = self.state.odom.yaw_rad
            heading_err = self._wrap_pi(
                math.atan2(entry.y - ry2, entry.x - rx2) - ryaw
            )
            if abs(heading_err) > math.radians(5):
                await self._run_precise_command(
                    angle_rad=heading_err, timeout_s=8.0,
                )

        self._set(progress=1.0, status=f"reached '{target}'")
        return SkillResult(
            success=True,
            message=f"reached '{target}' at ({entry.x:+.2f}, {entry.y:+.2f})",
            payload={
                "target_name": target,
                "object_x": entry.x,
                "object_y": entry.y,
                "object_confidence": entry.confidence,
                "object_hits": entry.hits,
            },
        )
