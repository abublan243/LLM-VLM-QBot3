"""
Go-To-Position — drive to (x, y) in the odom frame.

Two execution paths:
    use_precise_cmd=True (default): turn → drive → fine-turn using the
        Pi's closed-loop motion controller (/qbot3/precise_cmd).
    use_precise_cmd=False: continuous-velocity P controller using /qbot3/cmd_vel.

The precise-cmd path is more accurate but blocks per leg; the velocity path
is smoother and is the fallback when the motion controller is unreachable.

Tolerance / margin-of-error policy:
    * Default tolerance is 0.15 m — roughly half the QBot3 footprint, a
      sensible "close enough" for room-level navigation. Operators can
      tighten it for tight docking (``tolerance_m=0.05``) or loosen it
      for fast traversal (``tolerance_m=0.30``).
    * Three independent success conditions cover real-world odom drift:
        1. ``remaining < tolerance_m`` — the classic check.
        2. **Overshoot acceptance** — if a leg drives PAST the target so
           the heading flips by > 90° and we're within ``2 × tolerance``,
           we accept it as a success rather than hunting back.
        3. ``max_legs`` cap — refuses to drive forever if the encoder is
           lying about distance covered; returns a clear failure message.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from skills.base_skill import BaseSkill, SkillResult


class GoToPositionSkill(BaseSkill):
    name = "go_to_position"
    description = "Navigate to an (x, y) target in the odom frame."
    icon = "go_to_position"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target_x = float(params.get("x", 0.0))
        target_y = float(params.get("y", 0.0))
        tolerance = float(params.get("tolerance_m",
                                     self._defaults.get("tolerance_m", 0.15)))
        use_precise = bool(params.get("use_precise_cmd",
                                      self._defaults.get("use_precise_cmd", True)))
        max_legs = int(params.get("max_legs",
                                  self._defaults.get("max_legs", 12)))

        if use_precise:
            return await self._run_precise(target_x, target_y, tolerance, max_legs)
        return await self._run_velocity(target_x, target_y, tolerance)

    # ----- precise (closed-loop) path -----

    async def _run_precise(self, tx: float, ty: float, tol: float,
                           max_legs: int) -> SkillResult:
        self._set(progress=0.05, status="planning")
        heading_err = self._heading_error_to(tx, ty)
        distance = self._distance_to(tx, ty)
        if distance < tol:
            return SkillResult(success=True, message="already at target")

        # Step 1: turn to face target
        self._set(progress=0.2, status="turning to target")
        if abs(heading_err) > math.radians(5):
            ok = await self._run_precise_command(angle_rad=heading_err, timeout_s=15.0)
            if not ok:
                return SkillResult(success=False, message="turn-to-target failed")

        # Step 2: drive forward (one leg or two if it's far). The motion
        # controller's max single-leg distance is bounded by its 30 s
        # timeout — break the trip into ≤1.5 m legs for safety, recomputing
        # heading per leg.
        self._set(progress=0.5, status="driving")
        remaining = self._distance_to(tx, ty)
        overshoot_threshold = max(2.0 * tol, 0.30)   # accept overshoot within ~30 cm or 2×tol

        for leg_idx in range(max_legs):
            if self._aborted:
                return SkillResult(success=False, message="aborted")
            if remaining <= tol:
                break

            leg = min(1.5, remaining)
            ok = await self._run_precise_command(distance_m=leg, timeout_s=20.0)
            if not ok:
                return SkillResult(success=False, message="drive leg failed")

            # ---- Margin-of-error checks ----
            new_heading_err = self._heading_error_to(tx, ty)
            new_remaining = self._distance_to(tx, ty)

            # Overshoot: heading flipped >90° means we drove PAST the target.
            # If we're still within a reasonable margin, accept the leg.
            if abs(new_heading_err) > math.radians(90):
                if new_remaining <= overshoot_threshold:
                    return SkillResult(
                        success=True,
                        message=(f"reached ({tx:.2f}, {ty:.2f}) via overshoot "
                                 f"— off by {new_remaining:.2f} m"),
                    )
                # Otherwise turn around and continue
                await self._run_precise_command(
                    angle_rad=new_heading_err, timeout_s=10.0,
                )
            elif abs(new_heading_err) > math.radians(8):
                # Mild drift — re-align before the next leg
                await self._run_precise_command(
                    angle_rad=new_heading_err, timeout_s=10.0,
                )

            remaining = self._distance_to(tx, ty)
            self._set(progress=min(0.95, 1.0 - remaining / max(tol, 0.5)))
        else:
            return SkillResult(
                success=False,
                message=(f"max_legs={max_legs} exhausted; remaining "
                         f"{remaining:.2f} m to ({tx:.2f}, {ty:.2f})"),
            )

        return SkillResult(
            success=True,
            message=f"reached ({tx:.2f}, {ty:.2f}) within {tol:.2f} m",
        )

    # ----- velocity-loop fallback -----

    async def _run_velocity(self, tx: float, ty: float, tol: float) -> SkillResult:
        speed = 0.15
        Kp_ang = 1.4
        self._set(progress=0.05, status="driving (velocity)")
        while not self._aborted:
            await self._tick(0.05)
            distance = self._distance_to(tx, ty)
            if distance < tol:
                await self._stop_drive()
                return SkillResult(success=True, message=f"reached ({tx:.2f}, {ty:.2f})")
            heading_err = self._heading_error_to(tx, ty)
            angular_z = max(-1.0, min(1.0, Kp_ang * heading_err))
            # Slow down forward speed when we're not aimed correctly
            forward = speed * max(0.0, math.cos(heading_err))
            await self._drive(forward, angular_z)
            self._set(progress=max(0.05, 1.0 - distance / 5.0))
        return SkillResult(success=False, message="aborted")
