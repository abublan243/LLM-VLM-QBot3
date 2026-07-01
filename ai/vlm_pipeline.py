"""
VLM Pipeline — turns one camera frame + depth + task description into a
structured scene-understanding output the LLM planner can consume.

Stages (per project spec):
    1. YOLO detection on the RGB frame (Ultralytics YOLOv11).
    2. Depth analysis  (sensor_processor.compute_depth_stats).
    3. VLM API call   (OpenAI / Google / fusion / local-only).
    4. Structured VLMOutput  →  SharedState  →  Qt signal.

The pipeline is `run_async()`-driven; the AI mode handler awaits it under
qasync. Heavy work (YOLO inference, image encoding) is dispatched to a
thread pool so the GUI never blocks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ai.model_registry import MODELS, ModelInfo, Provider, get_model
from ai.yolo_world_detector import YoloWorldDetector
from ai.prompt_templates import SCENE_ANALYSIS_SYSTEM, SCENE_ANALYSIS_USER, READ_GAUGE_PROMPT
from core.sensor_processor import (
    DepthStats,
    compute_depth_stats,
    depth_at_pixel,
    deproject_pixel,
)
from core.shared_state import (
    CameraIntrinsics,
    Detection,
    SharedState,
    VLMOutput,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Result containers
# =====================================================================


class PipelineResult(dict):
    """The full structured output returned by VLMPipeline.run_async().

    Keys:
        detections (List[Detection])
        depth_stats (DepthStats | None)
        vlm_output  (VLMOutput)
        annotated_frame (np.ndarray BGR — original frame with YOLO overlays)
        latency_ms (float)
    """


# =====================================================================
# YOLO loader (singleton — loading the model takes ~3 s)
# =====================================================================


class _YoloRunner:
    _model = None
    _model_path: str = ""

    @classmethod
    def load(cls, weights: str = "yolo11l.pt") -> Any:
        if cls._model is not None and cls._model_path == weights:
            return cls._model
        logger.info("Loading YOLO weights: %s", weights)
        from ultralytics import YOLO
        cls._model = YOLO(weights)
        cls._model_path = weights
        return cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None


# =====================================================================
# VLMPipeline
# =====================================================================


class VLMPipeline(QObject):
    """End-to-end VLM step: YOLO + depth + remote model call."""

    yolo_loading = pyqtSignal(str)            # "loading"|"ready"
    yolo_ready = pyqtSignal()
    vlm_started = pyqtSignal(str)             # model name
    vlm_finished = pyqtSignal()

    def __init__(
        self,
        state: Optional[SharedState] = None,
        *,
        weights: str = "yolo11l.pt",
        confidence: float = 0.45,
        openai_key: str = "",
        google_key: str = "",
        max_tokens: int = 1024,
        performance_monitor: Optional[Any] = None,
        yolo_world_detector: Optional[YoloWorldDetector] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state or SharedState.instance()
        self.weights = weights
        self.confidence = confidence
        self.openai_key = openai_key
        self.google_key = google_key
        self.max_tokens = max_tokens
        self._perf = performance_monitor
        # Optional second-layer open-vocabulary detector (YOLO-World). Lets
        # the operator ask for arbitrary class names ("gun", "rifle",
        # "bottle", "person") that COCO may not cover or may miss.
        # Contributes Detections with `yw_` prefixed class names.
        self.yolo_world_detector = yolo_world_detector
        # Per-layer runtime enable flag for the COCO YOLO model itself.
        # YOLO-World carries its own .enabled flag on its detector object.
        # Two Vision-tab checkboxes let the operator drop a layer to
        # recover FPS without restarting or editing settings.json.
        self.coco_enabled: bool = True
        # Continuous detection task — populated by start_continuous_detection()
        self._continuous_task: Optional[asyncio.Task] = None

    # ---------------------------------------------------------------
    # Public configuration
    # ---------------------------------------------------------------

    def set_api_keys(self, openai_key: str, google_key: str) -> None:
        self.openai_key = openai_key or ""
        self.google_key = google_key or ""

    def set_confidence(self, conf: float) -> None:
        self.confidence = max(0.0, min(0.99, float(conf)))

    def set_layer_enabled(self, layer: str, enabled: bool) -> None:
        """Runtime toggle for one of the two detection layers.

        `layer` is one of: "coco" | "yolo_world" | "yw".
        Used by the Vision-tab checkboxes so the operator can drop a layer
        to recover FPS without restarting the app.
        """
        key = layer.lower().strip()
        on = bool(enabled)
        if key in ("coco", "yolo", "primary"):
            self.coco_enabled = on
            logger.info("VLM pipeline: COCO YOLO layer %s",
                        "ENABLED" if on else "DISABLED")
        elif key in ("yolo_world", "yw", "world"):
            if self.yolo_world_detector is not None:
                self.yolo_world_detector.set_enabled(on)
                logger.info("VLM pipeline: YOLO-World layer %s",
                            "ENABLED" if on else "DISABLED")
        else:
            logger.warning("Unknown detection layer key: %s", layer)

    def get_layer_states(self) -> Dict[str, bool]:
        """Return current enable state for the two layers — used by the
        Vision tab to initialise its checkboxes."""
        return {
            "coco": self.coco_enabled,
            "yolo_world": (
                self.yolo_world_detector.enabled
                if self.yolo_world_detector is not None else False
            ),
        }

    # ---------------------------------------------------------------
    # YOLO
    # ---------------------------------------------------------------

    async def ensure_yolo_loaded(self) -> None:
        if _YoloRunner.is_loaded():
            return
        self.yolo_loading.emit("loading")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _YoloRunner.load, self.weights)
        self.yolo_loading.emit("ready")
        self.yolo_ready.emit()

    def _run_yolo(self, frame_bgr: np.ndarray
                  ) -> Tuple[List[Detection], np.ndarray]:
        """Synchronous YOLO inference + annotated overlay. Runs in a worker thread.

        Runs up to two detection layers per call, each gated by an
        independent enable flag the Vision tab can toggle live:
          * COCO YOLO  (self.coco_enabled)
          * YOLO-World  (self.yolo_world_detector.enabled)
        Each disabled layer pays zero inference cost — that's the entire
        point of the runtime toggle, to recover FPS on demand.
        """
        detections: List[Detection] = []
        annotated = frame_bgr.copy()
        depth = self.state.depth_frame
        intr = self.state.camera_intrinsics
        ts = time.monotonic()

        # ---- Layer 1: COCO YOLO ----
        if not self.coco_enabled:
            # Skip primary inference entirely. Other layers still run.
            return self._maybe_run_extra_layers(
                detections, annotated, frame_bgr, depth, intr,
            )

        model = _YoloRunner.load(self.weights)
        results = model.predict(frame_bgr, conf=self.confidence, verbose=False)
        if self._perf is not None:
            self._perf.record_yolo_inference()

        if not results:
            return self._maybe_run_extra_layers(
                detections, annotated, frame_bgr, depth, intr,
            )
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        boxes = r.boxes
        if boxes is None:
            return self._maybe_run_extra_layers(
                detections, annotated, frame_bgr, depth, intr,
            )

        for b in boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            except Exception:
                continue
            x1, y1, x2, y2 = xyxy
            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2
            cls_name = names.get(cls_id, f"class_{cls_id}")

            dist_m = depth_at_pixel(depth, cx_px, cy_px) if depth is not None else 0.0
            pos_3d = deproject_pixel(intr, cx_px, cy_px, dist_m) if dist_m > 0 else None

            detections.append(Detection(
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
                centroid_xy=(cx_px, cy_px),
                distance_m=dist_m,
                position_3d=pos_3d,
                monotonic_ts=ts,
            ))

            # Overlay
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (108, 99, 255), 2)
            label = f"{cls_name} {conf:.2f}"
            if dist_m > 0:
                label += f" {dist_m:.2f}m"
            cv2.rectangle(annotated, (x1, y1 - 18), (x1 + 9 * len(label), y1), (108, 99, 255), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 245), 1, cv2.LINE_AA)

        return self._maybe_run_extra_layers(
            detections, annotated, frame_bgr, depth, intr,
        )

    def _maybe_run_extra_layers(
        self,
        detections: List[Detection],
        annotated: np.ndarray,
        frame_bgr: np.ndarray,
        depth: Any,
        intr: Any,
    ) -> Tuple[List[Detection], np.ndarray]:
        """Run the YOLO-World layer if its detector is attached AND its
        own `.enabled` flag is True. Disabled layer pays zero inference
        cost — that's the FPS recovery path.
        """
        # YOLO-World open-vocabulary — cyan boxes, yw_* class names.
        if (self.yolo_world_detector is not None
                and getattr(self.yolo_world_detector, "enabled", False)):
            yw = self.yolo_world_detector.detect(frame_bgr, depth, intr)
            if yw:
                detections.extend(yw)
                self.yolo_world_detector.draw_overlay(annotated, yw)

        return detections, annotated

    # ---------------------------------------------------------------
    # Continuous detection — keeps state.detected_objects fresh at all
    # times, regardless of mode. The Vision tab + SLAM viewer + skill
    # blocks all read from state.detected_objects, so YOLO needs to run
    # whether or not an AI task is currently planning.
    # ---------------------------------------------------------------

    async def start_continuous_detection(self, *, hz: float = 3.0) -> None:
        """Run YOLO on the latest RGB frame at `hz` Hz, forever, writing
        detections into SharedState. Idempotent: if already running, no-op.
        """
        if self._continuous_task is not None and not self._continuous_task.done():
            return
        loop = asyncio.get_running_loop()
        self._continuous_task = loop.create_task(self._continuous_loop(hz))

    def stop_continuous_detection(self) -> None:
        if self._continuous_task is not None and not self._continuous_task.done():
            self._continuous_task.cancel()
        self._continuous_task = None

    @property
    def is_continuous_running(self) -> bool:
        return self._continuous_task is not None and not self._continuous_task.done()

    async def _continuous_loop(self, hz: float) -> None:
        await self.ensure_yolo_loaded()
        period = 1.0 / max(0.5, float(hz))
        loop = asyncio.get_running_loop()
        last_frame_id = 0
        while True:
            try:
                with self.state.lock:
                    frame = (
                        None if self.state.rgb_frame is None
                        else self.state.rgb_frame.copy()
                    )
                    frame_ts = self.state.last_rgb_ts
                # Skip if no frame, or if we've already processed this exact frame
                # (the bridge writes a new frame each time it arrives, so ts changes).
                if frame is not None and frame_ts != last_frame_id:
                    last_frame_id = frame_ts
                    detections, _annot = await loop.run_in_executor(
                        None, self._run_yolo, frame,
                    )
                    self.state.set_detections(detections)
                    # Fold fresh detections into the persistent object-memory
                    # (RAG) so skills like go_to_object can look up class
                    # locations by name. Cheap: just a transform + dict merge.
                    mem = getattr(self.state, "object_memory", None)
                    if mem is not None and detections:
                        with self.state.lock:
                            pose = (
                                self.state.odom.x,
                                self.state.odom.y,
                                self.state.odom.yaw_rad,
                            )
                        try:
                            mem.update_from_detections(detections, pose)
                        except Exception as exc:
                            logger.warning("ObjectMemory update failed: %s", exc)
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                logger.info("Continuous YOLO loop cancelled")
                raise
            except Exception as exc:
                logger.warning("Continuous YOLO iteration failed: %s", exc)
                await asyncio.sleep(period)

    # ---------------------------------------------------------------
    # Pipeline entry-point
    # ---------------------------------------------------------------

    async def run_async(
        self,
        task_description: str,
        vlm_model_name: str,
        *,
        rgb_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Run the full VLM step. Returns PipelineResult and updates SharedState."""
        t0 = time.monotonic()

        # Frame snapshots (caller may pass; otherwise pull from SharedState)
        if rgb_frame is None:
            with self.state.lock:
                rgb_frame = None if self.state.rgb_frame is None else self.state.rgb_frame.copy()
        if depth_frame is None:
            with self.state.lock:
                depth_frame = None if self.state.depth_frame is None else self.state.depth_frame.copy()
        if rgb_frame is None:
            raise RuntimeError("VLMPipeline.run_async: no RGB frame available")

        # Stage 1: YOLO
        await self.ensure_yolo_loaded()
        loop = asyncio.get_running_loop()
        detections, annotated = await loop.run_in_executor(
            None, self._run_yolo, rgb_frame,
        )
        self.state.set_detections(detections)

        # Stage 2: depth analysis
        depth_stats = compute_depth_stats(depth_frame) if depth_frame is not None else None

        # Stage 3: VLM call
        info = get_model(vlm_model_name)
        self.vlm_started.emit(info.name)
        vlm_t0 = time.monotonic()
        try:
            scene_text, tokens = await self._dispatch_vlm(
                info, rgb_frame, task_description, detections, depth_stats,
            )
        except Exception as exc:
            logger.exception("VLM call failed: %s", exc)
            scene_text = (
                f"SCENE: VLM call failed ({type(exc).__name__}). "
                f"Falling back to YOLO + depth summary.\n"
                + self._yolo_only_summary(detections, depth_stats)
            )
            tokens = 0
        vlm_latency_ms = (time.monotonic() - vlm_t0) * 1000.0
        if self._perf is not None:
            self._perf.record_vlm_call(vlm_latency_ms, tokens=tokens)
        self.vlm_finished.emit()

        # Stage 4: structure & store
        sections = _parse_sections(scene_text)
        output = VLMOutput(
            scene_description=sections.get("SCENE", ""),
            object_relationships=sections.get("OBJECTS", ""),
            navigation_hints=sections.get("NAVIGATION", ""),
            task_observations=sections.get("TASK", ""),
            raw_text=scene_text,
            model=info.name,
            latency_ms=vlm_latency_ms,
            tokens_used=tokens,
            monotonic_ts=time.monotonic(),
        )
        self.state.set_vlm_output(output)

        return PipelineResult(
            detections=detections,
            depth_stats=depth_stats,
            vlm_output=output,
            annotated_frame=annotated,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    # ---------------------------------------------------------------
    # Provider dispatch
    # ---------------------------------------------------------------

    async def _dispatch_vlm(
        self,
        info: ModelInfo,
        rgb_frame: np.ndarray,
        task_description: str,
        detections: List[Detection],
        depth_stats: Optional[DepthStats],
    ) -> Tuple[str, int]:
        prompt_user = self._build_user_prompt(task_description, detections, depth_stats)

        if info.provider == Provider.LOCAL:
            return self._yolo_only_summary(detections, depth_stats), 0

        if info.provider == Provider.OPENAI:
            return await self._call_openai(info, rgb_frame, prompt_user)

        if info.provider == Provider.GOOGLE:
            return await self._call_google(info, rgb_frame, prompt_user)

        if info.provider == Provider.FUSION:
            return await self._call_fusion(info, rgb_frame, prompt_user)

        raise ValueError(f"Unknown VLM provider: {info.provider}")

    async def _call_openai(self, info: ModelInfo, frame: np.ndarray, user_prompt: str
                           ) -> Tuple[str, int]:
        if not self.openai_key:
            raise RuntimeError("OpenAI API key not configured")
        import openai
        b64 = _encode_jpeg_b64(frame)
        client = openai.AsyncOpenAI(api_key=self.openai_key)
        resp = await client.chat.completions.create(
            model=info.api_id,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SCENE_ANALYSIS_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    }},
                ]},
            ],
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    async def _call_google(self, info: ModelInfo, frame: np.ndarray, user_prompt: str
                           ) -> Tuple[str, int]:
        if not self.google_key:
            raise RuntimeError("Google API key not configured")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._call_google_sync, info, frame, user_prompt,
        )

    def _call_google_sync(self, info: ModelInfo, frame: np.ndarray, user_prompt: str
                          ) -> Tuple[str, int]:
        import google.generativeai as genai
        from PIL import Image
        genai.configure(api_key=self.google_key)
        model = genai.GenerativeModel(
            info.api_id,
            system_instruction=SCENE_ANALYSIS_SYSTEM,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        resp = model.generate_content(
            [user_prompt, pil_img],
            generation_config={"max_output_tokens": self.max_tokens, "temperature": 0.2},
        )
        try:
            text = resp.text or ""
        except Exception:
            text = str(resp)
        # google-generativeai exposes usage_metadata only on some endpoints
        tokens = 0
        try:
            tokens = int(getattr(resp, "usage_metadata", None).total_token_count)  # type: ignore[union-attr]
        except Exception:
            pass
        return text, tokens

    async def _call_fusion(self, info: ModelInfo, frame: np.ndarray, user_prompt: str
                           ) -> Tuple[str, int]:
        gpt_info = MODELS["gpt-4o"]
        gem_info = MODELS["gemini-1.5-pro"]
        results = await asyncio.gather(
            self._call_openai(gpt_info, frame, user_prompt),
            self._call_google(gem_info, frame, user_prompt),
            return_exceptions=True,
        )
        texts: List[str] = []
        tokens_total = 0
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Fusion member failed: %s", r)
                continue
            text, t = r
            texts.append(text)
            tokens_total += t
        if not texts:
            raise RuntimeError("Fusion: both VLM members failed")
        return _merge_text_sections(texts), tokens_total

    # ---------------------------------------------------------------
    # Prompt + fallback summary helpers
    # ---------------------------------------------------------------

    def _build_user_prompt(self, task: str, detections: List[Detection],
                           depth_stats: Optional[DepthStats]) -> str:
        det_payload = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": list(d.bbox_xyxy),
                "centroid": list(d.centroid_xy),
                "distance_m": round(d.distance_m, 2) if d.distance_m else None,
                "position_3d": d.position_3d,
            }
            for d in detections
        ]
        det_json = json.dumps(det_payload, indent=2) if det_payload else "[]"

        if depth_stats is None:
            depth_stats = DepthStats(
                valid_pixel_pct=0.0,
                nearest_distance_m=float("inf"),
                nearest_direction="center",
                free_corridor_width_px=0,
                sector_min_distances_m=[float("inf")] * 5,
                histogram_bins_m=[0] * 10,
                histogram_counts=[0] * 10,
            )

        snap = self.state.snapshot_for_planner()
        odom = snap["odom"]
        sector_str = ", ".join(f"{d:.2f}" if d != float("inf") else "inf"
                               for d in depth_stats.sector_min_distances_m)

        return SCENE_ANALYSIS_USER.format(
            task_description=task or "(no task — describe what you see)",
            yolo_detections_json=det_json,
            nearest_distance_m=depth_stats.nearest_distance_m
            if depth_stats.nearest_distance_m != float("inf") else 99.0,
            nearest_direction=depth_stats.nearest_direction,
            free_corridor_px=depth_stats.free_corridor_width_px,
            sector_distances=sector_str,
            valid_pixel_pct=depth_stats.valid_pixel_pct,
            pose_x=odom["x"], pose_y=odom["y"],
            pose_yaw_deg=np.degrees(odom["yaw_rad"]),
            battery_percent=snap["battery_percent"],
            bumpers_active=snap["bumpers_active"],
            cliff_active=snap["cliff_active"],
        )

    def _yolo_only_summary(self, detections: List[Detection],
                           depth_stats: Optional[DepthStats]) -> str:
        if detections:
            obj_lines = "\n".join(
                f"{d.class_name} (conf={d.confidence:.2f}, dist={d.distance_m:.2f}m)"
                for d in detections
            )
        else:
            obj_lines = "(no objects detected)"
        nearest = (
            f"{depth_stats.nearest_distance_m:.2f} m to the {depth_stats.nearest_direction}"
            if depth_stats and depth_stats.nearest_distance_m != float("inf")
            else "no obstacle within range"
        )
        return (
            f"SCENE: YOLO + depth summary (no remote VLM).\n"
            f"OBJECTS:\n{obj_lines}\n"
            f"NAVIGATION: nearest obstacle {nearest}.\n"
            f"TASK: relying on detector + depth only — VLM commentary unavailable."
        )

    # ---------------------------------------------------------------
    # Helper for the read_gauge skill
    # ---------------------------------------------------------------

    async def read_gauge_crop(self, crop_bgr: np.ndarray, *, vlm_model_name: str
                              ) -> Dict[str, Any]:
        """One-shot VLM call to read a gauge crop. Returns parsed JSON or {}."""
        info = get_model(vlm_model_name)
        if info.provider == Provider.OPENAI:
            text, _ = await self._call_openai(info, crop_bgr, READ_GAUGE_PROMPT)
        elif info.provider == Provider.GOOGLE:
            text, _ = await self._call_google(info, crop_bgr, READ_GAUGE_PROMPT)
        else:
            return {}
        try:
            return json.loads(_strip_json_fence(text))
        except Exception as exc:
            logger.warning("read_gauge JSON parse failed: %s", exc)
            return {"raw": text}


# =====================================================================
# Helpers
# =====================================================================


def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


_SECTION_KEYS = ("SCENE", "OBJECTS", "NAVIGATION", "TASK")


def _parse_sections(text: str) -> Dict[str, str]:
    """Pull SCENE/OBJECTS/NAVIGATION/TASK sections out of the VLM response."""
    out: Dict[str, str] = {k: "" for k in _SECTION_KEYS}
    if not text:
        return out
    lines = text.splitlines()
    current: Optional[str] = None
    buf: List[str] = []
    for raw in lines:
        line = raw.strip()
        matched = None
        for key in _SECTION_KEYS:
            if line.upper().startswith(key + ":"):
                matched = key
                break
        if matched is not None:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = matched
            after = line.split(":", 1)[1].strip() if ":" in line else ""
            buf = [after] if after else []
        else:
            if current is not None:
                buf.append(raw)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    if not any(out.values()):
        out["SCENE"] = text.strip()
    return out


def _merge_text_sections(texts: List[str]) -> str:
    """For fusion: concatenate per-section content, prefixing with model index."""
    sections = {k: [] for k in _SECTION_KEYS}
    for i, t in enumerate(texts):
        parsed = _parse_sections(t)
        for k in _SECTION_KEYS:
            if parsed.get(k):
                sections[k].append(f"[m{i+1}] {parsed[k]}")
    out_lines: List[str] = []
    for k in _SECTION_KEYS:
        if sections[k]:
            out_lines.append(f"{k}: " + " | ".join(sections[k]))
    return "\n".join(out_lines)


def _strip_json_fence(text: str) -> str:
    """Remove markdown code fences a model may include around JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text.lstrip("`")
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()
