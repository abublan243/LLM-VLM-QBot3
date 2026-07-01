"""
Return To Base — go to the named waypoint "base", which is saved by the
Manual mode "Save base" button (or via state.save_named_waypoint('base')).
"""

from __future__ import annotations

from typing import Any, Dict

from skills.base_skill import BaseSkill, SkillResult
from skills.go_to_position import GoToPositionSkill


class ReturnToBaseSkill(BaseSkill):
    name = "return_to_base"
    description = "Navigate to the saved 'base' waypoint."
    icon = "return_to_base"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        with self.state.lock:
            base = self.state.named_waypoints.get("base")
        if base is None:
            return SkillResult(success=False, message="no 'base' waypoint saved yet")

        bx, by = base[0], base[1]
        self._set(progress=0.1, status=f"heading to base ({bx:.2f}, {by:.2f})")

        # Pass the full skills_config so go_to_position picks up its real
        # defaults (was previously getting return_to_base's defaults under
        # the wrong key — meaningless tolerance / use_precise values).
        delegate = GoToPositionSkill(
            self.state, self.ros,
            skills_config=self._skills_config,
        )
        # Mirror our own pause/abort onto the delegate
        delegate._pause_event = self._pause_event
        delegate._aborted = False

        leg = await delegate.run({
            "x": bx,
            "y": by,
            "tolerance_m": float(params.get("position_tolerance_m",
                                            self._defaults.get("position_tolerance_m", 0.10))),
            "use_precise_cmd": True,
        })
        self._set(progress=1.0, status="done" if leg.success else "failed")
        return leg
