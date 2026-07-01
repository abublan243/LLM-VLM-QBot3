"""
Line Follower — HSV-threshold a line in the bottom band of the RGB frame and
steer toward its centroid. Runs until aborted or the line is lost.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from skills.base_skill import BaseSkill, SkillResult


class LineFollowerSkill(BaseSkill):
    name = "line_follower"
    description = "Follow a coloured line in the camera view (default = black)."
    icon = "line_follower"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        speed = float(params.get("speed_mps", 0.12))
        lower = np.array(params.get("line_color_hsv_lower", [0, 0, 0]), dtype=np.uint8)
        upper = np.array(params.get("line_color_hsv_upper", [180, 255, 60]), dtype=np.uint8)
        region_y_pct = float(params.get("camera_region_y_pct", 70)) / 100.0
        steering_gain = float(params.get("steering_gain", 0.004))
        deadband_px = float(params.get("steering_deadband_px", 12.0))

        miss_count = 0
        total_steps = 0
        self._set(progress=0.05, status="following line")

        while not self._aborted:
            await self._tick(0.05)

            with self.state.lock:
                frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            if frame is None:
                await self._tick(0.1)
                continue

            h, w = frame.shape[:2]
            band_y0 = int(h * region_y_pct)
            band = frame[band_y0:, :]
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            M = cv2.moments(mask)
            if M["m00"] < 200:        # no line
                miss_count += 1
                if miss_count > 30:    # ~1.5 s lost
                    await self._stop_drive()
                    return SkillResult(success=False, message="line lost")
                # Slow forward search while we look for it again
                await self._drive(speed * 0.4, 0.0)
                continue
            miss_count = 0

            cx = int(M["m10"] / M["m00"])
            error_px = cx - w // 2
            # Small deadband — line is broad enough that sub-deadband error
            # is just centroid jitter, not real lateral offset.
            if abs(error_px) < deadband_px:
                angular_z = 0.0
            else:
                angular_z = -steering_gain * error_px
                angular_z = max(-1.0, min(1.0, angular_z))

            await self._drive(speed, angular_z)

            total_steps += 1
            self._set(progress=min(0.95, 0.05 + (total_steps % 100) / 100.0 * 0.9))

        return SkillResult(success=True, message="line follow stopped")
