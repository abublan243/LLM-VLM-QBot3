"""
YOLO-World Detector — third (open-vocabulary) detection layer.

Where the COCO YOLO11 model is fixed to 80 classes and the military
detector is fixed to whatever its `.pt` was trained on, **YOLO-World**
accepts arbitrary class names as TEXT and detects them via CLIP-based
embedding matching. So you can ask it for "gun", "rifle", "soldier",
"helmet", "knife", "person with backpack" — anything — without
training.

Why it solves the operator's "VLM sees it but YOLO doesn't" problem:
- The COCO model misses small-at-distance threats because it was never
  trained on those classes at all.
- The dedicated military detector improves that, but is still bound by
  whatever scales its training set covered.
- YOLO-World uses the same backbone but trades raw closed-vocab accuracy
  for the ability to *try* any text query — including small / distant
  objects the LLM planner reads about in the VLM scene description.

This module follows the same pattern as `MilitaryDetector`:
- Lazy load, graceful no-op on missing weights / older Ultralytics.
- Returns plain `Detection` objects with a `yw_` class-name prefix so
  every downstream consumer (skills, planner, GUI overlay, RAG,
  mission reports) picks them up untouched.
- Distinct overlay colour (cyan) so the operator can tell the three
  detection layers apart in the Vision tab.

Weights:
- `yolov8s-world.pt` (~25 MB) is auto-downloaded by Ultralytics on
  first use, same as the COCO `yolo11l.pt` weights.
- Or larger: `yolov8m-world.pt`, `yolov8l-world.pt`, `yolov8x-worldv2.pt`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.sensor_processor import depth_at_pixel, deproject_pixel
from core.shared_state import CameraIntrinsics, Detection

logger = logging.getLogger(__name__)


# Default open-vocabulary classes — these are TEXT PROMPTS passed to the
# CLIP head, not model-internal class IDs. You can change them at any
# time (even at runtime via set_classes()) without touching the weights.
# Tuned for the operator's military-support use case + everyday humans.
DEFAULT_PROMPT_CLASSES: Tuple[str, ...] = (
    "person", "soldier", "helmet", "tactical helmet",
    "gun", "pistol", "handgun", "rifle", "assault rifle",
    "knife", "weapon",
    "backpack", "body armor",
    "grenade",
)


class YoloWorldDetector:
    """Open-vocabulary YOLO-World wrapper, drop-in for the existing
    Detection pipeline.
    """

    def __init__(
        self,
        weights: str = "models/yolov8s-world.pt",
        confidence: float = 0.25,
        class_prefix: str = "yw_",
        prompt_classes: Optional[Sequence[str]] = None,
        overlay_color_bgr: Tuple[int, int, int] = (200, 220, 0),  # cyan-ish
        enabled: bool = True,
    ) -> None:
        self.weights = weights
        self.confidence = float(confidence)
        self.class_prefix = class_prefix
        self.prompt_classes: List[str] = list(prompt_classes or DEFAULT_PROMPT_CLASSES)
        self.overlay_color = overlay_color_bgr
        self.enabled = bool(enabled)

        self._model: Any = None
        self._load_failed: bool = False
        self._classes_set: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled and self._load_failed and self._model is None:
            self._load_failed = False

    def set_confidence(self, conf: float) -> None:
        self.confidence = max(0.0, min(0.99, float(conf)))

    def set_classes(self, classes: Sequence[str]) -> None:
        """Change the open-vocabulary class list at runtime.

        Re-applies the CLIP class embeddings; takes ~200 ms. The planner
        can call this when the operator's task mentions a new keyword.
        """
        self.prompt_classes = list(classes)
        if self._model is not None:
            try:
                self._model.set_classes(self.prompt_classes)
                self._classes_set = True
                logger.info("YoloWorld: classes -> %s", self.prompt_classes)
            except Exception as exc:
                logger.warning("YoloWorld set_classes failed: %s", exc)

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.enabled or self._load_failed:
            return False
        try:
            # YOLOWorld lives under ultralytics.YOLOWorld in recent versions.
            try:
                from ultralytics import YOLOWorld
            except ImportError:
                logger.warning(
                    "YoloWorldDetector: this ultralytics version doesn't expose "
                    "YOLOWorld. Upgrade with `pip install -U ultralytics` to "
                    "enable open-vocabulary detection. Layer DISABLED."
                )
                self._load_failed = True
                return False

            t0 = time.monotonic()
            # If the weights path is given but missing, Ultralytics will
            # auto-download the model from its hub. We pass the path
            # straight in either way — same UX as the COCO YOLO loader.
            self._model = YOLOWorld(self.weights)
            # Set the prompt classes once at load time. Subsequent
            # set_classes() calls update them.
            self._model.set_classes(self.prompt_classes)
            self._classes_set = True
            logger.info(
                "YoloWorld: loaded '%s' (%d prompt classes) in %.1f ms",
                self.weights, len(self.prompt_classes),
                (time.monotonic() - t0) * 1000.0,
            )
            return True
        except Exception as exc:
            logger.exception("YoloWorld: failed to load '%s': %s",
                             self.weights, exc)
            self._load_failed = True
            return False

    def detect(
        self,
        frame_bgr: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[Detection]:
        if not self.enabled or not self._ensure_loaded():
            return []
        try:
            results = self._model.predict(
                frame_bgr, conf=self.confidence, verbose=False,
            )
        except Exception as exc:
            logger.warning("YoloWorld.predict failed: %s", exc)
            return []
        if not results:
            return []
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        boxes = r.boxes
        if boxes is None:
            return []

        ts = time.monotonic()
        out: List[Detection] = []
        for b in boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            except Exception:
                continue
            x1, y1, x2, y2 = xyxy
            raw_name = names.get(cls_id, f"class_{cls_id}")
            # Replace spaces with _ to keep skill-param parsing happy
            cls_name = f"{self.class_prefix}{raw_name.replace(' ', '_')}"
            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2

            dist_m = (
                depth_at_pixel(depth_frame, cx_px, cy_px)
                if depth_frame is not None else 0.0
            )
            pos_3d = (
                deproject_pixel(intrinsics, cx_px, cy_px, dist_m)
                if (intrinsics is not None and dist_m > 0) else None
            )

            out.append(Detection(
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
                centroid_xy=(cx_px, cy_px),
                distance_m=dist_m,
                position_3d=pos_3d,
                monotonic_ts=ts,
            ))
        return out

    def draw_overlay(
        self,
        annotated_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> np.ndarray:
        if not detections:
            return annotated_bgr
        color = self.overlay_color
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            if d.distance_m and d.distance_m > 0:
                label += f" {d.distance_m:.2f}m"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1,
            )
            cv2.rectangle(
                annotated_bgr,
                (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1),
                color, -1,
            )
            cv2.putText(
                annotated_bgr, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA,
            )
        return annotated_bgr
