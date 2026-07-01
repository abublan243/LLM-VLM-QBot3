"""
explore_room — semantic active search for a target object.

Replaces the legacy `search_object` 360-spin with something the
graduation report can actually call "intelligent": at each step the
skill consults FOUR information sources and combines them into a
target-belief probability grid that is co-registered with the SLAM
occupancy grid. It then drives the robot to the highest-scoring
reachable cell, scans, updates the belief from what it just saw, and
repeats.

Information used (combined inside ai/target_belief_map.py):
  1. VLM scene text — if the latest VLM output flagged
     `VLM-sees-target` plus a bearing (center / left / right / ...),
     the belief in that cone is boosted.
  2. YOLO detection list — every cycle without a matching detection
     decays belief inside the camera FOV cone (we LOOKED and didn't
     see it).
  3. SLAM occupancy + frontier cells — only reachable free cells are
     candidate goals; the boundary between free and unknown
     (frontiers) is where the target could still plausibly be hiding.
  4. Visited trajectory — every spot the robot has driven through
     gets its belief decayed (we've been there).

Exit conditions:
  * success: a matching detection appears in `state.detected_objects`
    above `min_confidence` — the caller should next invoke
    `approach_object` for the precise stop.
  * failure: max iterations reached, max wall-clock reached, or no
    reachable frontier candidate remains (belief grid exhausted).

Safety:
  * Bumper / cliff / wheel-drop interrupt every drive leg (handled
    by the underlying go_to_position delegate).
  * Always publishes zero cmd_vel on exit.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from ai.target_belief_map import TargetBeliefMap
from skills.base_skill import BaseSkill, SkillResult
from skills.go_to_position import GoToPositionSkill

logger = logging.getLogger(__name__)


class ExploreRoomSkill(BaseSkill):
    name = "explore_room"
    description = (
        "Semantic active search: drive to the highest-belief reachable "
        "frontier and scan, repeating until the target detection arrives "
        "or the room is exhausted. Uses VLM scene text, YOLO detections, "
        "the SLAM occupancy grid, and the robot's own trajectory to "
        "pick where to look next. Use this for non-trivial rooms in "
        "place of the legacy 360-spin search_object."
    )
    icon = "search_object"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_class", "")).strip().lower()
        if not target:
            return SkillResult(success=False, message="no target_class given")

        target_keywords: List[str] = [
            str(k).strip().lower()
            for k in (params.get("target_keywords") or [])
            if str(k).strip()
        ]

        max_iterations = int(params.get("max_iterations", 8))
        max_duration_s = float(params.get("max_duration_s", 120.0))
        min_confidence = float(params.get("min_confidence", 0.30))
        scan_after_arrival = bool(params.get("scan_after_arrival", True))
        scan_half_rad = math.radians(float(params.get("scan_half_deg", 60.0)))
        leg_tolerance_m = float(params.get("leg_tolerance_m", 0.25))
        min_goal_distance_m = float(params.get("min_goal_distance_m", 0.5))
        max_goal_distance_m = float(params.get("max_goal_distance_m", 4.0))

        slam = getattr(self.state, "slam_manager", None)
        if slam is None:
            return SkillResult(
                success=False,
                message="SLAM manager unavailable — explore_room needs the "
                        "host-side grid to pick goals",
            )

        # Build a belief grid aligned with the SLAM grid so cell indices
        # match without any per-cycle transform.
        belief = TargetBeliefMap(
            size_cells=slam.size,
            resolution_m=slam.resolution,
            origin_cell=slam._origin_cell,    # private, intentional — they're sister classes
        )

        t_start = time.monotonic()
        self._set(progress=0.02, status=f"exploring for '{target}'")

        # Early-exit shortcut: maybe the target is already visible.
        match = self._find_matching_detection(target, target_keywords, min_confidence)
        if match is not None:
            return SkillResult(
                success=True,
                message=f"already detected '{match[0]}'",
                payload={"target_class": match[0], "confidence": match[1]},
            )

        last_iter_msg = ""
        for it in range(1, max_iterations + 1):
            if self._aborted:
                return SkillResult(success=False, message="aborted")
            if time.monotonic() - t_start > max_duration_s:
                return SkillResult(
                    success=False,
                    message=f"max_duration_s ({max_duration_s:.0f}s) elapsed "
                            f"after {it - 1} legs ({last_iter_msg})",
                )

            with self.state.lock:
                rx = self.state.odom.x
                ry = self.state.odom.y
                ryaw = self.state.odom.yaw_rad
                intr = self.state.camera_intrinsics
                vlm = self.state.vlm_last_output
            robot_xy = (rx, ry)

            # ---- 1. Fold belief updates from this cycle's signals ----
            fov_half_rad = (
                math.atan(intr.width / (2.0 * intr.fx))
                if intr.is_valid() and intr.fx > 0 else math.radians(35.0)
            )
            belief.update_yolo_negative(
                robot_xy, ryaw,
                fov_half_rad=fov_half_rad,
                max_range_m=4.0,
            )
            belief.update_visited(robot_xy, radius_m=0.4)

            vlm_bearing, vlm_distance, vlm_conf = self._parse_vlm_hint(vlm, target)
            if vlm_bearing is not None:
                belief.update_vlm_bearing(
                    robot_xy, ryaw, vlm_bearing,
                    distance_band=vlm_distance or "mid",
                    confidence=vlm_conf or 0.6,
                )

            # ---- 2. Reachability via SLAM ----
            frontiers = slam.get_frontier_cells()
            free_mask = slam.get_free_space_map()
            belief.apply_frontier_prior(frontiers, weight=0.5)

            # ---- 3. Pick the next goal ----
            goal = belief.next_goal(
                robot_xy, free_mask, frontiers,
                prefer_frontiers=True,
                min_goal_distance_m=min_goal_distance_m,
                max_goal_distance_m=max_goal_distance_m,
            )
            if goal is None:
                last_iter_msg = "no reachable frontier candidate"
                logger.info("explore_room: %s — giving up", last_iter_msg)
                return SkillResult(
                    success=False,
                    message=f"exhausted reachable belief (iter {it})",
                )

            logger.info(
                "explore_room iter %d/%d → goal (%.2f, %.2f) belief=%.2f d=%.2fm",
                it, max_iterations, goal.world_xy[0], goal.world_xy[1],
                goal.belief, goal.distance_m,
            )
            last_iter_msg = (
                f"goal=({goal.world_xy[0]:.2f},{goal.world_xy[1]:.2f}) "
                f"belief={goal.belief:.2f}"
            )
            self._set(
                progress=0.05 + 0.85 * (it / max_iterations),
                status=f"iter {it}/{max_iterations}: drive to "
                       f"({goal.world_xy[0]:+.2f},{goal.world_xy[1]:+.2f})",
            )

            # ---- 4. Drive there via go_to_position ----
            mover = GoToPositionSkill(
                self.state, self.ros, skills_config=self._skills_config,
            )
            drive_params = {
                "x": float(goal.world_xy[0]),
                "y": float(goal.world_xy[1]),
                "tolerance_m": leg_tolerance_m,
                "max_legs": 6,
            }
            result = await mover.run(drive_params)
            if self._aborted:
                return SkillResult(success=False, message="aborted")
            if not result.success:
                # Couldn't reach this goal — strongly decay belief there
                # so the next selection doesn't loop on it.
                cr, cc = goal.cell_rc
                belief.log_belief[cr, cc] -= 3.0
                logger.info(
                    "explore_room: leg failed (%s) — penalising goal cell",
                    result.message,
                )

            # ---- 5. Check for target after the leg ----
            match = self._find_matching_detection(target, target_keywords, min_confidence)
            if match is not None:
                return SkillResult(
                    success=True,
                    message=f"found '{match[0]}' at iter {it}",
                    payload={"target_class": match[0], "confidence": match[1]},
                )

            # ---- 6. Optional scan — slow pivot to widen the FOV ----
            if scan_after_arrival:
                await self._scan_in_place(scan_half_rad)
                match = self._find_matching_detection(target, target_keywords, min_confidence)
                if match is not None:
                    return SkillResult(
                        success=True,
                        message=f"found '{match[0]}' during scan at iter {it}",
                        payload={"target_class": match[0], "confidence": match[1]},
                    )

        return SkillResult(
            success=False,
            message=f"exhausted {max_iterations} iterations without finding '{target}' "
                    f"({last_iter_msg})",
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _find_matching_detection(
        self,
        target: str,
        keywords: List[str],
        min_confidence: float,
    ) -> Optional[Tuple[str, float]]:
        with self.state.lock:
            dets = list(self.state.detected_objects)
        best: Optional[Tuple[str, float]] = None
        for d in dets:
            if d.confidence < min_confidence:
                continue
            cls = d.class_name.lower()
            if cls == target or target in cls or any(k in cls for k in keywords):
                if best is None or d.confidence > best[1]:
                    best = (d.class_name, d.confidence)
        return best

    @staticmethod
    def _parse_vlm_hint(
        vlm_output: Any,
        target: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Extract (bearing, distance_band, confidence) from the VLM
        scene text. The planner-prompt rule forces the VLM to emit
        `VLM-sees-target <bearing> <distance_band>` when it perceives
        the target but YOLO does not.

        Returns (None, None, None) if no parsable hint."""
        if vlm_output is None:
            return None, None, None
        text = ""
        for attr in ("task_observations", "raw_text"):
            text = getattr(vlm_output, attr, "") or ""
            if text:
                break
        if not text:
            return None, None, None
        low = text.lower()
        if target not in low and "vlm-sees-target" not in low:
            # No mention of the target → no usable hint.
            return None, None, None
        if "vlm-sees-target" not in low:
            return None, None, None
        # Extract bearing + distance keywords near the marker.
        chunk = low.split("vlm-sees-target", 1)[1][:80]
        bearing = None
        for b in ("slight_left", "slight_right", "left", "right", "center"):
            if b in chunk:
                bearing = b
                break
        distance = None
        for d in ("near", "mid", "far"):
            if d in chunk:
                distance = d
                break
        return bearing, distance, 0.7

    async def _scan_in_place(self, half_rad: float) -> None:
        """Pivot ±half_rad around the current heading, slowly, so YOLO
        and the next belief-update get a wider view from this pose."""
        steps = 4
        per_step = (2.0 * half_rad) / steps
        # Turn left, then turn right past centre — we approximate with
        # two short angular bursts via cmd_vel to keep this skill
        # self-contained (no precise_cmd round-trip per step).
        for direction in (+1, -1):
            target_dt = abs(per_step) / 0.6   # ~0.6 rad/s pivot
            t_end = time.monotonic() + target_dt
            while time.monotonic() < t_end:
                await self._drive(0.0, 0.6 * direction)
                await self._tick(0.05)
            await self._drive(0.0, 0.0)
            await self._tick(0.4)         # let YOLO catch up
