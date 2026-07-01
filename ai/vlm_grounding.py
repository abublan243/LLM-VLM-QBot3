"""
VLM visual ROI grounding — close the YOLO blind-spot by asking a VLM
to point at the operator's target in the live RGB frame and return a
strict-JSON box / point.

Used by the new `vlm_reach` skill when:
  * the operator's task describes a clear target (e.g. "approach the
    bottle"), AND
  * the VLM scene text says it sees the target, BUT
  * the YOLO/YOLO-World detection list contains no matching box.

The grounding is a two-tier cascade for cost control:

  1. **gpt-4o-mini** — fast and cheap. Strict JSON schema with five
     fields:
        target_visible  (bool)
        confidence      (float, 0..1)
        bbox_norm       ([x1, y1, x2, y2] in 0..1 image coords, or null)
        point_norm      ([x, y] in 0..1, or null)
        bearing         ("center"|"left"|"right"|"slight_left"|"slight_right")
     If `confidence >= mini_confidence_floor`, return that result.

  2. **gpt-4o** (escalation) — only invoked when the mini call returns
     low confidence OR can't lock a point. Same schema, better at small
     objects.

The returned `GroundingResult` carries both the normalized 2-D pixel
coordinates AND the deprojected 3-D position (camera optical frame)
when depth + intrinsics are available, so the caller can plug it
straight into a navigation goal.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.sensor_processor import depth_at_pixel, deproject_pixel
from core.shared_state import CameraIntrinsics

logger = logging.getLogger(__name__)


# Strict prompt: ANY deviation from the schema breaks downstream parsing.
GROUNDING_SYSTEM = (
    "You are a visual grounding assistant for a mobile robot. You will "
    "be given one camera frame and a target object name. Return ONLY a "
    "JSON object — no markdown, no prose, no code fences. Schema:\n"
    "{\n"
    '  "target_visible": <true|false>,\n'
    '  "confidence": <number 0..1>,\n'
    '  "bbox_norm": [<x1>, <y1>, <x2>, <y2>] | null,\n'
    '  "point_norm": [<x>, <y>] | null,\n'
    '  "bearing": "center" | "left" | "right" | "slight_left" | "slight_right"\n'
    "}\n"
    "All coordinates are NORMALISED 0..1 (top-left origin). `point_norm` "
    "must be the centre of the target. `bearing` is the target's "
    "horizontal direction relative to the camera centre. If you cannot "
    "see the target, set target_visible=false, confidence=0, both "
    "bbox_norm and point_norm to null."
)


@dataclass
class GroundingResult:
    """Outcome of a grounding call."""

    target_visible: bool
    confidence: float
    bbox_norm: Optional[Tuple[float, float, float, float]]  # x1, y1, x2, y2
    point_norm: Optional[Tuple[float, float]]               # x, y
    bearing: str
    # Pixel & depth — populated by `enrich_with_depth` after the LLM call.
    point_pixel: Optional[Tuple[int, int]] = None
    distance_m: float = 0.0
    position_3d_optical: Optional[Tuple[float, float, float]] = None
    model_used: str = ""
    latency_ms: float = 0.0
    escalated: bool = False
    raw_text: str = ""

    @property
    def usable(self) -> bool:
        """True if downstream code can drive to a 3-D goal."""
        return (
            self.target_visible
            and self.point_pixel is not None
            and self.distance_m > 0.0
            and self.position_3d_optical is not None
        )


class VlmGrounding:
    """Two-tier visual grounding pipeline (gpt-4o-mini → gpt-4o)."""

    MINI_MODEL = "gpt-4o-mini"
    HEAVY_MODEL = "gpt-4o"

    def __init__(
        self,
        *,
        openai_key: str,
        mini_confidence_floor: float = 0.55,
        max_tokens: int = 300,
        jpeg_quality: int = 80,
    ) -> None:
        self.openai_key = openai_key or ""
        self.mini_confidence_floor = float(mini_confidence_floor)
        self.max_tokens = int(max_tokens)
        self.jpeg_quality = int(jpeg_quality)

    def set_api_key(self, key: str) -> None:
        self.openai_key = key or ""

    # ---------------------------------------------------------------
    # Public entry
    # ---------------------------------------------------------------

    async def ground(
        self,
        rgb_bgr: np.ndarray,
        target_name: str,
        *,
        depth_frame: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> GroundingResult:
        """Run the cascade. Returns a GroundingResult; never raises on
        API failures — falls back to a target_visible=False result so
        the caller can degrade gracefully."""
        if not self.openai_key:
            return GroundingResult(
                target_visible=False, confidence=0.0,
                bbox_norm=None, point_norm=None, bearing="center",
                raw_text="no openai key configured",
            )
        if rgb_bgr is None or rgb_bgr.size == 0:
            return GroundingResult(
                target_visible=False, confidence=0.0,
                bbox_norm=None, point_norm=None, bearing="center",
                raw_text="no frame",
            )

        t0 = time.monotonic()

        # ---- Tier 1: gpt-4o-mini ----
        result = await self._call(self.MINI_MODEL, rgb_bgr, target_name)
        result.latency_ms = (time.monotonic() - t0) * 1000.0
        result.model_used = self.MINI_MODEL

        needs_escalation = (
            not result.target_visible
            or result.confidence < self.mini_confidence_floor
            or result.point_norm is None
        )
        if needs_escalation:
            # ---- Tier 2: gpt-4o ----
            t1 = time.monotonic()
            heavy = await self._call(self.HEAVY_MODEL, rgb_bgr, target_name)
            heavy.latency_ms = (time.monotonic() - t1) * 1000.0
            heavy.model_used = self.HEAVY_MODEL
            heavy.escalated = True
            # Trust the heavy model unless it returned worse — common
            # for very small targets where mini might luck onto a hit.
            if (
                heavy.target_visible
                and (heavy.point_norm is not None or heavy.bbox_norm is not None)
                and heavy.confidence >= max(0.0, result.confidence - 0.1)
            ):
                result = heavy
            else:
                # Keep the mini result but flag that we escalated.
                result.escalated = True

        self._enrich_with_depth(result, rgb_bgr.shape, depth_frame, intrinsics)
        return result

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    async def _call(
        self,
        model_id: str,
        rgb_bgr: np.ndarray,
        target_name: str,
    ) -> GroundingResult:
        try:
            import openai
        except Exception as exc:
            logger.warning("openai SDK unavailable for grounding: %s", exc)
            return GroundingResult(
                target_visible=False, confidence=0.0,
                bbox_norm=None, point_norm=None, bearing="center",
                raw_text=f"openai import failed: {exc}",
            )

        try:
            b64 = self._encode_jpeg_b64(rgb_bgr)
            client = openai.AsyncOpenAI(api_key=self.openai_key)
            resp = await client.chat.completions.create(
                model=model_id,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                temperature=0.0,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": f"Target object: {target_name}\n"
                                 f"Return the JSON described in the system "
                                 f"prompt for this frame."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        }},
                    ]},
                ],
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Grounding call to %s failed: %s", model_id, exc)
            return GroundingResult(
                target_visible=False, confidence=0.0,
                bbox_norm=None, point_norm=None, bearing="center",
                raw_text=f"{type(exc).__name__}: {exc}",
            )

        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> GroundingResult:
        """Parse the JSON response. The strict schema makes this
        deterministic, but we still tolerate stray code fences for
        robustness across providers."""
        raw = text or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3].strip()
        try:
            data: Dict[str, Any] = json.loads(cleaned)
        except Exception:
            # Last-ditch attempt: grab the outermost {...}
            try:
                m = re.search(r"\{[\s\S]*\}", cleaned)
                if m is None:
                    raise ValueError("no json object found")
                data = json.loads(m.group(0))
            except Exception as exc:
                logger.debug("Grounding JSON parse failed: %s — raw=%r", exc, raw[:200])
                return GroundingResult(
                    target_visible=False, confidence=0.0,
                    bbox_norm=None, point_norm=None, bearing="center",
                    raw_text=raw,
                )

        target_visible = bool(data.get("target_visible", False))
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        bbox_norm = _coerce_box(data.get("bbox_norm"))
        point_norm = _coerce_point(data.get("point_norm"))
        bearing = str(data.get("bearing", "center")).strip().lower()
        if bearing not in (
            "center", "left", "right", "slight_left", "slight_right",
        ):
            bearing = "center"

        # If point_norm wasn't given but a box was, use the box centre.
        if point_norm is None and bbox_norm is not None:
            cx = 0.5 * (bbox_norm[0] + bbox_norm[2])
            cy = 0.5 * (bbox_norm[1] + bbox_norm[3])
            point_norm = (cx, cy)

        return GroundingResult(
            target_visible=target_visible,
            confidence=max(0.0, min(1.0, confidence)),
            bbox_norm=bbox_norm,
            point_norm=point_norm,
            bearing=bearing,
            raw_text=raw,
        )

    @staticmethod
    def _enrich_with_depth(
        result: GroundingResult,
        frame_shape: Tuple[int, int, int],
        depth_frame: Optional[np.ndarray],
        intrinsics: Optional[CameraIntrinsics],
    ) -> None:
        if not result.target_visible or result.point_norm is None:
            return
        h, w = frame_shape[:2]
        px = int(round(result.point_norm[0] * (w - 1)))
        py = int(round(result.point_norm[1] * (h - 1)))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        result.point_pixel = (px, py)
        if depth_frame is None or intrinsics is None or not intrinsics.is_valid():
            return
        # Use a windowed median lookup so a single zero pixel in the
        # depth frame doesn't kill us.
        d = depth_at_pixel(depth_frame, px, py)
        if d <= 0.0:
            return
        result.distance_m = float(d)
        result.position_3d_optical = deproject_pixel(intrinsics, px, py, d)

    @staticmethod
    def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 80) -> str:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed for grounding payload")
        return base64.b64encode(buf.tobytes()).decode("ascii")


def _coerce_point(p: Any) -> Optional[Tuple[float, float]]:
    if p is None:
        return None
    try:
        x, y = float(p[0]), float(p[1])
    except Exception:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return (x, y)


def _coerce_box(b: Any) -> Optional[Tuple[float, float, float, float]]:
    if b is None:
        return None
    try:
        x1, y1, x2, y2 = (float(b[i]) for i in range(4))
    except Exception:
        return None
    x1 = max(0.0, min(1.0, x1))
    x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)
