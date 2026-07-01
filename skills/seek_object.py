"""
Seek Object — close the distance when the VLM can see a target but YOLO
hasn't latched on yet ("move to get more confidence").

Use case: the VLM (GPT-4o / Gemini) reports "there is a bottle on the
table" in the scene description, but no detector layer (COCO YOLO11,
YOLO-World) has produced a `Detection` for it yet — typically because
the object is too small in the frame at current range. This skill drives
forward at low speed, optionally biased toward a VLM-reported bearing
("center", "left", "right"), and polls `state.detected_objects` every
cycle. As soon as the target_class shows up above `min_confidence`, the
skill exits success and the planner should follow up with `approach_object`
for the precise stop.

Safety guards:
- Hard distance cap (`max_distance_m`, default 3.0 m) — won't run forever.
- Hard time cap (`max_duration_s`, default 30 s).
- Bumper / cliff / wheel-drop interrupt the moment any fires.
- Depth-based obstacle stop (`min_obstacle_distance_m`, default 0.30 m).
- Always publishes zero cmd_vel on exit.

Parameters:
- `target_class` (REQUIRED) — name to look for in `state.detected_objects`.
  Accepts prefixed names from any layer: `person`, `yw_gun`, `bottle`.
  The planner should pass whatever the VLM mentioned, including prefixes.
- `target_keywords` (optional) — list of additional substrings to match
  against any detection class. Useful when the planner isn't 100% sure
  which prefixed form will appear (e.g. ["gun", "pistol", "rifle"]).
- `bearing_hint` (optional) — "center" | "left" | "right". Lets the
  planner pass the VLM's spatial cue so the seek isn't purely forward.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from skills.base_skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class SeekObjectSkill(BaseSkill):
    name = "seek_object"
    description = (
        "Drive forward (with optional bearing bias) while polling for a "
        "target detection. Use when the VLM scene text mentions an object "
        "but no detector has produced a YOLO/YOLO-World box for it yet — "
        "typically because the object is too far away. Exits as soon as a "
        "matching detection arrives; the planner should then call "
        "approach_object for the precise stop."
    )
    icon = "approach_object"

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_class", "")).strip().lower()
        keywords_raw = params.get("target_keywords") or []
        if isinstance(keywords_raw, str):
            keywords_raw = [keywords_raw]
        keywords: List[str] = [str(k).strip().lower() for k in keywords_raw if str(k).strip()]

        if not target and not keywords:
            return SkillResult(
                success=False,
                message="seek_object needs target_class or target_keywords",
            )

        approach_speed = float(params.get("approach_speed_mps", 0.12))
        min_confidence = float(params.get("min_confidence", 0.30))
        max_distance_m = float(params.get("max_distance_m", 3.0))
        max_duration_s = float(params.get("max_duration_s", 30.0))
        min_obstacle_m = float(params.get("min_obstacle_distance_m", 0.30))
        bearing_hint = str(params.get("bearing_hint", "center")).strip().lower()

        # Map VLM bearing words to a small angular bias. The planner
        # should re-issue the skill if the bearing changes; we don't try
        # to do mid-skill VLM steering here (those calls are slow).
        bearing_angular = {
            "center":      0.0,
            "centre":      0.0,
            "front":       0.0,
            "left":       +0.25,    # CCW
            "slight_left":+0.15,
            "right":      -0.25,    # CW
            "slight_right":-0.15,
        }.get(bearing_hint, 0.0)

        # Capture starting pose so we can enforce the distance cap.
        with self.state.lock:
            start_x = self.state.odom.x
            start_y = self.state.odom.y

        t_start = time.monotonic()
        self._set(progress=0.05, status=f"seeking '{target or keywords[0]}'")
        logger.info(
            "seek_object: target='%s' keywords=%s bearing=%s speed=%.2f "
            "max_d=%.1fm max_t=%.0fs",
            target, keywords, bearing_hint, approach_speed,
            max_distance_m, max_duration_s,
        )

        while not self._aborted:
            await self._tick(0.10)

            # ---- 1. Safety: bumpers / cliff / wheel-drop ----
            with self.state.lock:
                bump = self.state.bumpers.any_active()
                cliff = self.state.cliff.any_active()
                drop = self.state.wheel_drop.any_active()
            if bump:
                await self._stop_drive()
                return SkillResult(
                    success=False,
                    message="bumper fired during seek — stopped"
                )
            if cliff:
                await self._stop_drive()
                return SkillResult(
                    success=False,
                    message="cliff sensor fired during seek — stopped"
                )
            if drop:
                await self._stop_drive()
                return SkillResult(
                    success=False,
                    message="wheel drop fired during seek — stopped"
                )

            # ---- 2. Did the target finally show up in detections? ----
            with self.state.lock:
                dets = list(self.state.detected_objects)
            hit = _match_detection(dets, target, keywords, min_confidence)
            if hit is not None:
                await self._stop_drive()
                self._set(
                    progress=1.0,
                    status=f"acquired '{hit.class_name}' conf={hit.confidence:.2f}",
                )
                return SkillResult(
                    success=True,
                    message=(
                        f"target acquired: class={hit.class_name} "
                        f"conf={hit.confidence:.2f} "
                        f"dist={hit.distance_m:.2f}m — hand off to approach_object"
                    ),
                    payload={
                        "matched_class": hit.class_name,
                        "confidence": hit.confidence,
                        "distance_m": hit.distance_m,
                    },
                )

            # ---- 3. Distance / time caps ----
            with self.state.lock:
                cur_x = self.state.odom.x
                cur_y = self.state.odom.y
            dx = cur_x - start_x
            dy = cur_y - start_y
            traveled = (dx * dx + dy * dy) ** 0.5
            elapsed = time.monotonic() - t_start
            if traveled >= max_distance_m:
                await self._stop_drive()
                return SkillResult(
                    success=False,
                    message=(
                        f"travelled {traveled:.2f} m without acquiring "
                        f"'{target or keywords}' — try repositioning or "
                        f"adjusting target_keywords"
                    ),
                )
            if elapsed >= max_duration_s:
                await self._stop_drive()
                return SkillResult(
                    success=False,
                    message=f"timed out after {elapsed:.0f}s without acquisition",
                )

            # ---- 4. Depth-based obstacle check (host-side) ----
            with self.state.lock:
                depth = (
                    None if self.state.depth_frame is None
                    else self.state.depth_frame
                )
            if depth is not None:
                # Cheap centre-strip median in metres
                h, w = depth.shape[:2]
                strip = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
                valid = strip[strip > 0]
                if valid.size > 100:
                    nearest = float(valid.min()) / 1000.0  # uint16 mm → m
                    if nearest < min_obstacle_m:
                        await self._stop_drive()
                        return SkillResult(
                            success=False,
                            message=(
                                f"obstacle at {nearest:.2f} m (< "
                                f"{min_obstacle_m:.2f} m) — cannot proceed"
                            ),
                        )

            # ---- 5. Drive forward (with optional bearing bias) ----
            await self._drive(approach_speed, bearing_angular)
            # Progress reflects distance fraction covered (0.05 → 0.95)
            frac = min(0.9, traveled / max(0.1, max_distance_m))
            self._set(progress=0.05 + 0.9 * frac,
                      status=f"seeking '{target or keywords[0]}'  "
                             f"{traveled:.2f}/{max_distance_m:.1f}m")

        await self._stop_drive()
        return SkillResult(success=False, message="aborted")


def _match_detection(
    detections: List[Any],
    target: str,
    keywords: List[str],
    min_confidence: float,
) -> Optional[Any]:
    """Return the highest-confidence detection that matches the target.

    Matching rules:
      1. Exact class_name == target (case-insensitive).
      2. target appears anywhere as a substring of class_name (handles
         prefixed names: `yw_pistol` matches target=`pistol`).
      3. Any keyword appears as substring in class_name.
    """
    if not detections:
        return None
    best = None
    best_conf = min_confidence
    for d in detections:
        if d.confidence < min_confidence:
            continue
        name_lower = d.class_name.lower()
        matched = False
        if target:
            if name_lower == target or target in name_lower:
                matched = True
        if not matched and keywords:
            if any(kw in name_lower for kw in keywords):
                matched = True
        if matched and d.confidence >= best_conf:
            best = d
            best_conf = d.confidence
    return best
