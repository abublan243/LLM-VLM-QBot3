"""
Wall Follower — PD controller maintaining a target distance from a chosen wall.

Reads the depth frame from SharedState; uses the rightmost (or leftmost)
columns to estimate distance to the wall on that side. PD on the error.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from skills.base_skill import BaseSkill, SkillResult


class WallFollowerSkill(BaseSkill):
    name = "wall_follower"
    description = "Maintain a fixed distance from a wall using depth — runs until aborted."
    icon = "wall_follower"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target_distance = float(params.get("target_distance_m", 0.25))
        speed = float(params.get("speed_mps", 0.15))
        wall_side = str(params.get("wall_side", "right")).lower()
        kp = float(params.get("kp", 1.5))
        kd = float(params.get("kd", 0.4))
        col_width = int(params.get("depth_column_width_px", 80))

        # ROS angular_z convention: positive = CCW = LEFT turn.
        # When following the RIGHT wall, "too far" (positive error) must turn
        # the robot RIGHT (CW = negative angular_z) to close in. So sign is
        # NEGATIVE for right-wall follow, POSITIVE for left.
        sign = -1.0 if wall_side == "right" else 1.0
        prev_error = 0.0
        last_valid = 0
        total_steps = 0

        self._set(progress=0.05, status=f"following {wall_side} wall")

        while not self._aborted:
            await self._tick(0.05)

            with self.state.lock:
                depth = self.state.depth_frame
            if depth is None:
                await self._tick(0.1)
                continue

            h, w = depth.shape
            col_width = min(col_width, max(8, w // 4))
            if wall_side == "right":
                strip = depth[h // 3: 2 * h // 3, w - col_width:]
            else:
                strip = depth[h // 3: 2 * h // 3, :col_width]

            valid = strip[(strip > 100) & (strip < 4000)]
            if valid.size < 50:
                # Lost the wall — slow turn TOWARD the wall to re-acquire
                await self._drive(speed * 0.5, sign * 0.5)
                last_valid += 1
                if last_valid > 60:        # ~3 seconds without a wall
                    return SkillResult(success=False, message="wall lost — no valid depth")
                continue
            last_valid = 0

            measured = float(np.median(valid)) / 1000.0
            error = measured - target_distance
            d_error = error - prev_error
            prev_error = error

            # PD: positive error means too far → turn toward wall
            angular_z = sign * (kp * error + kd * d_error)
            angular_z = max(-1.0, min(1.0, angular_z))

            await self._drive(speed, angular_z)

            total_steps += 1
            # No fixed end — just keep ticking progress so the GUI bar pulses
            self._set(progress=min(0.95, 0.05 + (total_steps % 100) / 100.0 * 0.9))

        return SkillResult(success=True, message="wall follow stopped")
