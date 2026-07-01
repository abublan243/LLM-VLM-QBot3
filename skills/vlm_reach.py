"""
vlm_reach — reach an object the VLM can see but YOLO cannot.

Pipeline per iteration:
  1. **Ground** — call the VLM grounding cascade (gpt-4o-mini, escalating
     to gpt-4o when confidence is low) on the live RGB. We get back a
     normalised image point + bbox + bearing + confidence.
  2. **Deproject** — convert the image point to a 3-D position using
     the current depth frame and camera intrinsics. The robot's
     base_link X/Y then comes from the optical → base → world chain.
  3. **Safe viewpoint** — DON'T drive into the target. Compute a
     viewpoint `stop_distance_m` short of the target along the
     robot→target line, then snap it to the nearest known-free cell on
     the SLAM grid (via slam.is_free / cell search). Refuse to move if
     no safe cell exists.
  4. **Drive** — delegate to `go_to_position` to navigate there.
  5. **Verify** — recapture; if YOLO now has a matching box OR a fresh
     grounding call returns confidence above the success floor at close
     range, return success. Otherwise repeat up to `max_iterations`.

Required parameter: `target_name` (the object the operator wants).
Optional knobs cover stop distance, confidence floors, viewpoint
snapping radius, and iteration caps.

Why a separate skill from `seek_object`? `seek_object` drives blindly
forward; `vlm_reach` plans a goal using the VLM's actual visual
grounding, then validates that goal against the SLAM map before
moving. It's the planner's preferred tool when the VLM has spotted
the target but YOLO has not.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

from skills.base_skill import BaseSkill, SkillResult
from skills.go_to_position import GoToPositionSkill

logger = logging.getLogger(__name__)


class VlmReachSkill(BaseSkill):
    name = "vlm_reach"
    description = (
        "Reach an object the VLM can see but YOLO cannot. Uses a "
        "GPT-4o-mini → GPT-4o visual-grounding cascade to extract a "
        "precise image point for the target, deprojects it to a 3-D "
        "world position via the live depth frame, picks a SAFE "
        "viewpoint short of the target by snapping to a known-free "
        "cell on the SLAM grid, then drives there via go_to_position. "
        "Re-verifies at each hop. Use this when the VLM clearly "
        "perceives the target but no YOLO detection exists for it."
    )
    icon = "approach_object"

    def __init__(
        self,
        state,
        ros,
        *,
        skills_config: Optional[Dict[str, Any]] = None,
        vlm_grounding=None,
    ) -> None:
        super().__init__(state, ros, skills_config=skills_config)
        # Injected at construction time by mode_ai / mode_skills (same
        # special-case pattern as read_gauge). May be None if the
        # operator's openai key isn't configured — we'll return a clean
        # failure in that case rather than crashing.
        self._grounding = vlm_grounding

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_name", "")).strip()
        if not target:
            return SkillResult(success=False, message="no target_name given")

        if self._grounding is None or not getattr(self._grounding, "openai_key", ""):
            return SkillResult(
                success=False,
                message="vlm_reach unavailable — VLM grounding not configured "
                        "(missing OpenAI key or grounding service).",
            )

        max_iterations = int(params.get("max_iterations", 4))
        max_duration_s = float(params.get("max_duration_s", 90.0))
        stop_distance_m = float(params.get("stop_distance_m", 0.50))
        viewpoint_snap_m = float(params.get("viewpoint_snap_m", 0.40))
        success_confidence = float(params.get("success_confidence", 0.55))
        leg_tolerance_m = float(params.get("leg_tolerance_m", 0.20))

        slam = getattr(self.state, "slam_manager", None)
        if slam is None:
            return SkillResult(
                success=False,
                message="SLAM manager unavailable — vlm_reach needs free-space "
                        "info to pick a safe viewpoint",
            )

        t_start = time.monotonic()
        self._set(progress=0.02, status=f"grounding '{target}'")

        last_message = ""
        for it in range(1, max_iterations + 1):
            if self._aborted:
                return SkillResult(success=False, message="aborted")
            if time.monotonic() - t_start > max_duration_s:
                return SkillResult(
                    success=False,
                    message=f"max_duration_s ({max_duration_s:.0f}s) elapsed "
                            f"({last_message})",
                )

            # ---- 1. Snapshot the live frame + sensors ----
            with self.state.lock:
                rgb = (
                    None if self.state.rgb_frame is None
                    else self.state.rgb_frame.copy()
                )
                depth = (
                    None if self.state.depth_frame is None
                    else self.state.depth_frame.copy()
                )
                intr = self.state.camera_intrinsics
                rx = self.state.odom.x
                ry = self.state.odom.y
                ryaw = self.state.odom.yaw_rad
            if rgb is None:
                last_message = "no RGB frame yet"
                await self._tick(0.2)
                continue

            # ---- 2. Ground (mini → 4o cascade) ----
            self._set(
                progress=0.10 + 0.85 * ((it - 1) / max_iterations),
                status=f"iter {it}/{max_iterations}: grounding '{target}'",
            )
            ground = await self._grounding.ground(
                rgb, target, depth_frame=depth, intrinsics=intr,
            )
            logger.info(
                "vlm_reach iter %d: visible=%s conf=%.2f model=%s "
                "esc=%s point=%s dist=%.2fm",
                it, ground.target_visible, ground.confidence,
                ground.model_used, ground.escalated,
                ground.point_pixel, ground.distance_m,
            )

            # ---- 3. Success check: high-confidence + close range ----
            if (
                ground.target_visible
                and ground.confidence >= success_confidence
                and 0.0 < ground.distance_m <= stop_distance_m + 0.15
            ):
                return SkillResult(
                    success=True,
                    message=f"reached '{target}' (conf={ground.confidence:.2f}, "
                            f"d={ground.distance_m:.2f}m)",
                    payload={
                        "target_name": target,
                        "confidence": ground.confidence,
                        "model": ground.model_used,
                        "distance_m": ground.distance_m,
                    },
                )

            # If grounding failed entirely, give the perception loop a
            # beat and retry from a slightly different pose. A short
            # ±15° pivot moves the target off any RGB occlusion edge.
            if not ground.usable:
                last_message = (
                    f"grounding insufficient ({ground.model_used} "
                    f"conf={ground.confidence:.2f})"
                )
                if it < max_iterations:
                    await self._nudge_view()
                continue

            # ---- 4. Project to a world XY target ----
            target_xy = self._optical_3d_to_world(
                ground.position_3d_optical, rx, ry, ryaw,
            )
            if target_xy is None:
                last_message = "no usable 3-D projection from depth"
                continue
            tx, ty = target_xy

            # ---- 5. Safe viewpoint: stop_distance_m short of the target ----
            viewpoint = self._safe_viewpoint(
                (rx, ry), (tx, ty),
                stop_distance_m=stop_distance_m,
                snap_radius_m=viewpoint_snap_m,
                slam=slam,
            )
            if viewpoint is None:
                last_message = (
                    f"no safe viewpoint near target ({tx:+.2f},{ty:+.2f})"
                )
                logger.info("vlm_reach: %s", last_message)
                if it < max_iterations:
                    await self._nudge_view()
                continue
            vx, vy = viewpoint
            logger.info(
                "vlm_reach iter %d: target=(%.2f,%.2f) → viewpoint=(%.2f,%.2f)",
                it, tx, ty, vx, vy,
            )
            last_message = (
                f"drive to viewpoint ({vx:+.2f},{vy:+.2f}) "
                f"for target ({tx:+.2f},{ty:+.2f})"
            )

            # ---- 6. Drive there ----
            mover = GoToPositionSkill(
                self.state, self.ros, skills_config=self._skills_config,
            )
            res = await mover.run({
                "x": vx,
                "y": vy,
                "tolerance_m": leg_tolerance_m,
                "max_legs": 6,
            })
            if self._aborted:
                return SkillResult(success=False, message="aborted")
            if not res.success:
                last_message = f"navigation to viewpoint failed: {res.message}"
                logger.info("vlm_reach: %s", last_message)
                # Don't give up — next iteration will re-ground from the
                # new pose and the SLAM grid may have learned the
                # blocking obstacle on the way in.
                continue

            # ---- 7. Cheap YOLO check after the leg ----
            ydet = self._matching_yolo_detection(target)
            if ydet is not None:
                return SkillResult(
                    success=True,
                    message=f"YOLO acquired '{ydet[0]}' (conf={ydet[1]:.2f}) "
                            f"after vlm_reach leg {it}",
                    payload={
                        "target_name": target,
                        "yolo_class": ydet[0],
                        "yolo_confidence": ydet[1],
                    },
                )

        return SkillResult(
            success=False,
            message=f"exhausted {max_iterations} iterations ({last_message})",
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _optical_3d_to_world(
        opt_xyz: Tuple[float, float, float],
        rx: float, ry: float, ryaw: float,
    ) -> Optional[Tuple[float, float]]:
        """Camera-optical (X right, Y down, Z forward) → world XY."""
        if opt_xyz is None:
            return None
        try:
            x_cam, _y_cam, z_cam = opt_xyz
        except Exception:
            return None
        if z_cam <= 0.0:
            return None
        # base_link (X forward, Y left)
        x_base = z_cam
        y_base = -x_cam
        c = math.cos(ryaw); s = math.sin(ryaw)
        wx = rx + c * x_base - s * y_base
        wy = ry + s * x_base + c * y_base
        return wx, wy

    @staticmethod
    def _safe_viewpoint(
        robot_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        *,
        stop_distance_m: float,
        snap_radius_m: float,
        slam,
    ) -> Optional[Tuple[float, float]]:
        """Pick a (vx, vy) that sits `stop_distance_m` short of the
        target along the robot→target line, snapped to the nearest
        known-free SLAM cell within `snap_radius_m`."""
        rx, ry = robot_xy
        tx, ty = target_xy
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None
        # Step back from the target along the robot→target direction.
        ux = dx / dist
        uy = dy / dist
        ideal_x = tx - ux * stop_distance_m
        ideal_y = ty - uy * stop_distance_m
        # If the ideal viewpoint is already free, we're done.
        if slam.is_free(ideal_x, ideal_y):
            return (ideal_x, ideal_y)
        # Search outward in 5 cm rings up to snap_radius_m for a free cell.
        snap_cells = max(1, int(round(snap_radius_m / slam.resolution)))
        cr0, cc0 = slam._origin_cell
        target_cr = int(round(cr0 + ideal_y / slam.resolution))
        target_cc = int(round(cc0 + ideal_x / slam.resolution))
        for ring in range(1, snap_cells + 1):
            for d_r in range(-ring, ring + 1):
                for d_c in range(-ring, ring + 1):
                    if abs(d_r) != ring and abs(d_c) != ring:
                        continue
                    r = target_cr + d_r
                    c = target_cc + d_c
                    if not (0 <= r < slam.size and 0 <= c < slam.size):
                        continue
                    wx, wy = slam.cell_to_world(r, c)
                    if slam.is_free(wx, wy):
                        return (wx, wy)
        return None

    def _matching_yolo_detection(
        self, target: str,
    ) -> Optional[Tuple[str, float]]:
        low = target.lower()
        with self.state.lock:
            dets = list(self.state.detected_objects)
        best: Optional[Tuple[str, float]] = None
        for d in dets:
            cls = d.class_name.lower()
            if cls == low or low in cls:
                if best is None or d.confidence > best[1]:
                    best = (d.class_name, d.confidence)
        return best

    async def _nudge_view(self) -> None:
        """Small ±15° pivot to break out of an unfortunate occlusion."""
        for direction in (+1, -1):
            t_end = time.monotonic() + math.radians(15.0) / 0.6
            while time.monotonic() < t_end:
                await self._drive(0.0, 0.6 * direction)
                await self._tick(0.05)
            await self._drive(0.0, 0.0)
            await self._tick(0.3)
