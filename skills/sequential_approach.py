"""
Sequential Approach — visit a list of object classes in order. For each:
search → approach → log position → continue.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from skills.approach_object import ApproachObjectSkill
from skills.base_skill import BaseSkill, SkillResult
from skills.search_object import SearchObjectSkill


class SequentialApproachSkill(BaseSkill):
    name = "sequential_approach"
    description = "Visit a list of object classes in order (search + approach each)."
    icon = "sequential_approach"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        sequence: List[str] = list(params.get("target_sequence") or [])
        if not sequence:
            return SkillResult(success=False, message="empty target_sequence")

        stop_distance = float(params.get("stop_distance_m",
                                         self._defaults.get("stop_distance_m", 0.5)))
        per_target_timeout = float(params.get("per_target_timeout_s",
                                              self._defaults.get("per_target_timeout_s", 60)))

        visited: List[Dict[str, Any]] = []

        for i, target in enumerate(sequence):
            if self._aborted:
                return SkillResult(success=False, message="aborted",
                                   payload={"visited": visited})
            self._set(
                progress=i / len(sequence),
                status=f"target {i+1}/{len(sequence)}: '{target}'",
            )

            # Search — propagate skills_config so search_object picks up its
            # real defaults (rotation_speed, max_full_rotations, etc.).
            search = SearchObjectSkill(
                self.state, self.ros,
                skills_config=self._skills_config,
            )
            search._pause_event = self._pause_event
            search_res = await search.run({"target_class": target})
            if not search_res.success:
                visited.append({
                    "target": target, "found": False, "approach_success": False,
                    "message": search_res.message,
                })
                continue

            # Approach — propagate skills_config so the new oscillation
            # knobs (deadband, smoothing) take effect inside the delegate.
            approach = ApproachObjectSkill(
                self.state, self.ros,
                skills_config=self._skills_config,
            )
            approach._pause_event = self._pause_event

            t_start = time.monotonic()
            approach_res = await approach.run({
                "target_class": target,
                "stop_distance_m": stop_distance,
            })
            elapsed = time.monotonic() - t_start

            with self.state.lock:
                pose = (self.state.odom.x, self.state.odom.y, self.state.odom.yaw_rad)

            visited.append({
                "target": target,
                "found": True,
                "approach_success": approach_res.success,
                "message": approach_res.message,
                "pose_at_arrival": pose,
                "elapsed_s": round(elapsed, 1),
            })
            self.state.append_event(
                "INFO",
                f"{self.name}: {target} -> {'OK' if approach_res.success else 'FAIL'} "
                f"@({pose[0]:.2f},{pose[1]:.2f})",
            )

            if elapsed > per_target_timeout:
                return SkillResult(
                    success=False,
                    message=f"timeout on '{target}'",
                    payload={"visited": visited},
                )

        all_ok = all(v.get("approach_success") for v in visited)
        return SkillResult(
            success=all_ok,
            message=f"visited {len(visited)} targets",
            payload={"visited": visited},
        )
