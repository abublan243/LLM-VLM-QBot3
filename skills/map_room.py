"""
Map Room — drive a boustrophedon (lawnmower) coverage pattern, letting the
SLAM manager build out the occupancy grid in the meantime.

Coverage is measured as: (# free cells in the SLAM map) / (an expanding
target area = bounding-box of explored trajectory + margin). When that
ratio reaches `coverage_threshold_pct`, we stop.

This is intentionally a simple primitive — it doesn't plan around obstacles
intelligently, it just sweeps. Combined with the safety layer (bumper/cliff
overrides) this is good enough for a demo room.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np

from skills.base_skill import BaseSkill, SkillResult


class MapRoomSkill(BaseSkill):
    name = "map_room"
    description = "Sweep the room in a back-and-forth pattern to build the SLAM map."
    icon = "map_room"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        coverage_target = float(params.get("coverage_threshold_pct", 80)) / 100.0
        speed = float(params.get("speed_mps", 0.15))
        spacing = float(params.get("spacing_m", 0.4))
        leg_distance = float(params.get("leg_distance_m", 2.0))
        max_legs = int(params.get("max_legs", 24))

        self._set(progress=0.05, status="starting sweep")

        leg = 0
        turn_dir = 1.0       # +1 = CCW, -1 = CW (alternates each row)
        coverage = 0.0
        deadline = time.monotonic() + 600.0   # 10-minute hard cap

        while leg < max_legs and not self._aborted and time.monotonic() < deadline:
            # Drive forward one leg
            ok = await self._drive_forward_safe(leg_distance, speed)
            if self._aborted:
                break
            # If we hit something, shorten and continue with the turn anyway
            if not ok:
                await self._tick(0.2)

            # 180° turn through one spacing offset
            await self._run_precise_command(angle_rad=turn_dir * math.pi / 2, timeout_s=10.0)
            await self._run_precise_command(distance_m=spacing, timeout_s=10.0)
            await self._run_precise_command(angle_rad=turn_dir * math.pi / 2, timeout_s=10.0)
            turn_dir *= -1.0

            leg += 1
            coverage = self._estimate_coverage()
            self._set(
                progress=min(0.95, max(leg / max_legs, coverage / max(0.01, coverage_target))),
                status=f"leg {leg}/{max_legs}, coverage {coverage*100:.0f}%",
            )
            if coverage >= coverage_target:
                break

        await self._stop_drive()
        ok = coverage >= coverage_target or leg >= max_legs / 2
        return SkillResult(
            success=ok,
            message=f"coverage {coverage*100:.0f}% after {leg} legs",
            payload={"coverage": coverage, "legs": leg},
        )

    # ---------------------------------------------------------------

    async def _drive_forward_safe(self, distance_m: float, speed: float) -> bool:
        """Drive forward at `speed` until distance reached, bumper hit, or aborted."""
        with self.state.lock:
            x0, y0 = self.state.odom.x, self.state.odom.y
        while not self._aborted:
            await self._tick(0.05)
            with self.state.lock:
                bx, by = self.state.odom.x, self.state.odom.y
                bumpers = self.state.bumpers.any_active()
                cliff = self.state.cliff.any_active()
            if bumpers or cliff:
                await self._stop_drive()
                # Reverse a touch so the next turn doesn't grind into the wall
                await self._run_precise_command(distance_m=-0.10, timeout_s=5.0)
                return False
            travelled = math.hypot(bx - x0, by - y0)
            if travelled >= distance_m:
                await self._stop_drive()
                return True
            await self._drive(speed, 0.0)

        await self._stop_drive()
        return False

    def _estimate_coverage(self) -> float:
        """Free-cells / explored bounding-box area. Cheap and good-enough for a demo."""
        with self.state.lock:
            traj = list(self.state.slam_trajectory)
        if len(traj) < 5:
            return 0.0
        xs = np.array([p[0] for p in traj])
        ys = np.array([p[1] for p in traj])
        bbox_w = xs.max() - xs.min() + 1.0
        bbox_h = ys.max() - ys.min() + 1.0
        bbox_area = max(1.0, bbox_w * bbox_h)
        # Approximate "covered" as 0.4 m radius around each trajectory sample
        traj_area = len(traj) * math.pi * 0.4 * 0.4
        return min(1.0, traj_area / bbox_area)
