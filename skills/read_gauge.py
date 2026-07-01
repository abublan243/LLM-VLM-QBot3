"""
Read Gauge — approach an object of class "gauge" (or whatever the operator
specified), capture a high-res crop of its bounding box, and ask the VLM
to read the displayed value.

Returns the parsed JSON {"value": ..., "units": ..., "confidence": ...}
in the SkillResult.payload, and logs the reading to the event log so it
shows up in the Task Log panel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.shared_state import Detection
from skills.approach_object import ApproachObjectSkill
from skills.base_skill import BaseSkill, SkillResult


class ReadGaugeSkill(BaseSkill):
    name = "read_gauge"
    description = "Approach a gauge/dial/meter and read the value with the VLM."
    icon = "read_gauge"

    def __init__(self, state, ros, *,
                 vlm_pipeline: Any = None, vlm_model_name: str = "gpt-4o",
                 skills_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(state, ros, skills_config=skills_config)
        self._vlm = vlm_pipeline
        self._vlm_model = vlm_model_name

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        target = str(params.get("target_class", "gauge")).lower()
        approach_distance = float(params.get("approach_distance_m", 0.6))
        crop_padding = int(params.get("crop_padding_px", 40))

        # Step 1: approach the gauge
        self._set(progress=0.05, status=f"approaching '{target}'")
        # Pass through the full skills_config so the delegate's approach_object
        # gets its real defaults (deadband, smoothing, etc.) rather than the
        # read_gauge defaults under the wrong key.
        approach = ApproachObjectSkill(
            self.state, self.ros,
            skills_config=self._skills_config,
        )
        approach._pause_event = self._pause_event
        result = await approach.run({
            "target_class": target,
            "stop_distance_m": approach_distance,
        })
        if not result.success:
            return SkillResult(success=False, message=f"approach failed: {result.message}")

        # Step 2: capture a crop
        self._set(progress=0.6, status="capturing gauge crop")
        crop, det = self._best_crop(target, crop_padding)
        if crop is None:
            return SkillResult(success=False, message="no gauge in view at stopping distance")

        # Step 3: VLM read
        if self._vlm is None:
            return SkillResult(success=False, message="VLM pipeline not provided")
        self._set(progress=0.8, status="asking VLM")
        try:
            reading = await self._vlm.read_gauge_crop(crop, vlm_model_name=self._vlm_model)
        except Exception as exc:
            return SkillResult(success=False, message=f"VLM call failed: {exc}")

        value = reading.get("value")
        units = reading.get("units", "")
        msg = f"gauge reading: {value} {units}".strip()
        self.state.append_event("INFO", msg)
        self._set(progress=1.0, status="read complete")
        return SkillResult(success=value is not None, message=msg, payload=reading)

    # ---------------------------------------------------------------

    def _best_crop(self, target: str, pad: int):
        with self.state.lock:
            frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
            dets = list(self.state.detected_objects)
        if frame is None:
            return None, None
        candidates = [d for d in dets if d.class_name.lower() == target]
        if not candidates:
            return None, None
        det: Detection = max(candidates, key=lambda d: d.confidence)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox_xyxy
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return frame[y1:y2, x1:x2].copy(), det
