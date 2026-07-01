"""
Approach Object — align heading to a detected object's centroid X, then drive
forward until its measured depth drops below `stop_distance_m`.

Anti-oscillation design:
    * Bbox center is EMA-smoothed (cuts YOLO jitter at the source).
    * A pixel-deadband around frame-center suppresses tiny corrections — if
      the target is "centered enough", angular_z is zeroed and the robot
      drives straight.
    * Turn-then-drive: when off-axis by more than `turn_only_px`, forward
      speed is zeroed so we rotate cleanly into alignment before advancing
      (prevents the parallax-feedback oscillator).
    * A confidence floor rejects spurious low-confidence detections that
      would otherwise yank the controller every few frames.

Deceleration profile:
    Forward speed eases from `approach_speed_mps` down to
    `min_approach_speed_mps` across `slow_radius_m` ahead of the stop band,
    using a cubic smoothstep (3t² − 2t³). Soft entry into the slow zone
    and a soft exit at the stop distance — no jerk at either end.

Assumes the YOLO/VLM pipeline keeps SharedState.detected_objects fresh.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.shared_state import Detection
from skills.base_skill import BaseSkill, SkillResult


class ApproachObjectSkill(BaseSkill):
    name = "approach_object"
    description = "Centre on a detected object and drive up to it."
    icon = "approach_object"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_class", "")).lower()
        if not target:
            return SkillResult(success=False, message="no target_class given")

        stop_distance = float(params.get("stop_distance_m", 0.5))
        alignment_gain = float(params.get("alignment_gain", 0.003))
        speed = float(params.get("approach_speed_mps", 0.15))
        lost_timeout_s = float(params.get("lost_target_timeout_s", 2.0))
        # Anti-oscillation knobs (defaults tuned for 640-px wide RGB):
        deadband_px = float(params.get("alignment_deadband_px", 25.0))
        turn_only_px = float(params.get("turn_only_threshold_px", 60.0))
        smoothing = float(params.get("alignment_smoothing", 0.7))   # EMA α on the previous estimate
        max_angular = float(params.get("max_angular_radps", 0.6))
        min_confidence = float(params.get("min_confidence", 0.35))
        # Deceleration profile — smoothstep ramp over `slow_radius_m` ahead of
        # `stop_distance`. The robot is at full speed beyond the slow zone and
        # eases to `min_approach_speed_mps` as it reaches the stop band.
        slow_radius = float(params.get("slow_radius_m", 0.6))
        min_approach_speed = float(params.get("min_approach_speed_mps", 0.04))

        last_seen_ts = time.monotonic()
        cx_smooth: Optional[float] = None
        self._set(progress=0.05, status=f"approaching '{target}'")

        while not self._aborted:
            await self._tick(0.05)
            det = self._best_detection(target, min_confidence)

            if det is None:
                if time.monotonic() - last_seen_ts > lost_timeout_s:
                    await self._stop_drive()
                    return SkillResult(success=False, message="target lost")
                # Brief loss: hold our last bearing rather than spinning a
                # search arc — that arc was a major contributor to the
                # left-right wobble whenever YOLO blinked.
                await self._drive(0.0, 0.0)
                continue
            last_seen_ts = time.monotonic()

            # Stop test up front — avoids one extra cmd publish after arrival
            if 0.05 < det.distance_m <= stop_distance:
                await self._stop_drive()
                self._set(progress=1.0, status="reached")
                return SkillResult(
                    success=True,
                    message=f"reached '{target}' at {det.distance_m:.2f} m",
                    payload={"target_class": target, "distance_m": det.distance_m},
                )

            frame_w = self._frame_width()
            cx_raw = float(det.centroid_xy[0])

            # EMA the centroid so YOLO jitter doesn't ride straight into
            # angular_z. Reseed on first frame so we don't drift in from 0.
            if cx_smooth is None:
                cx_smooth = cx_raw
            else:
                cx_smooth = smoothing * cx_smooth + (1.0 - smoothing) * cx_raw

            error_px = cx_smooth - frame_w / 2.0
            abs_error = abs(error_px)

            # Deadband: ~3-4° of HFOV at 640 px is well below what the robot
            # can resolve mechanically. Don't fight it.
            if abs_error < deadband_px:
                angular_z = 0.0
            else:
                angular_z = -alignment_gain * error_px
                if angular_z > max_angular:
                    angular_z = max_angular
                elif angular_z < -max_angular:
                    angular_z = -max_angular

            # Turn-before-drive: when the target is way off-axis, just
            # rotate. Driving forward at the same time changes the bbox
            # center via parallax and would re-feed the alignment loop.
            if abs_error > turn_only_px:
                forward_speed = 0.0
            else:
                forward_speed = speed
                # Smoothstep deceleration: ease from `speed` to
                # `min_approach_speed` across the slow zone ahead of the
                # stop band. Cubic ease (3t² − 2t³) gives a soft entry and
                # a soft exit — no jerk at either end, unlike the old
                # linear-with-floor ramp.
                if det.distance_m > 0.05 and slow_radius > 1e-3:
                    progress = (det.distance_m - stop_distance) / slow_radius
                    if progress < 1.0:
                        t = max(0.0, min(1.0, progress))
                        ease = t * t * (3.0 - 2.0 * t)
                        forward_speed = min_approach_speed + (
                            speed - min_approach_speed
                        ) * ease

            await self._drive(forward_speed, angular_z)

            self._set(
                progress=min(0.95, max(0.1, 1.0 - (det.distance_m / 3.0))),
                status=f"approaching '{target}' ({det.distance_m:.2f} m)",
            )

        return SkillResult(success=False, message="aborted")

    def _best_detection(self, target: str, min_confidence: float
                        ) -> Optional[Detection]:
        with self.state.lock:
            candidates = [
                d for d in self.state.detected_objects
                if d.class_name.lower() == target
                and d.confidence >= min_confidence
            ]
        if not candidates:
            return None
        # Highest confidence wins; ties broken by closest
        candidates.sort(key=lambda d: (-d.confidence, d.distance_m or 999))
        return candidates[0]

    def _frame_width(self) -> int:
        with self.state.lock:
            if self.state.rgb_frame is not None:
                return int(self.state.rgb_frame.shape[1])
            if self.state.camera_intrinsics.is_valid():
                return int(self.state.camera_intrinsics.width)
        return 640
