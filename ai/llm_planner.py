"""
LLM Planner — turns task description + VLM output + robot state into a
validated action JSON {action_type, low_level_command|skill_command, status, …}.

Phases:
    1. Context assembly  — pulls a snapshot from SharedState + VLMOutput.
    2. LLM call          — OpenAI / Google / fusion. JSON mode where supported.
    3. JSON parse        — strict; falls back to a "blocked" stop on failure.
    4. Safety layer      — clamps velocities, overrides on bumper/cliff/battery.

Public entry point:
    plan_async(task_description, vlm_output, model_name,
               execution_type, allowed_skills) -> LLMOutput
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ai.model_registry import MODELS, ModelInfo, Provider, get_model
from ai.prompt_templates import (
    ROBOT_PLANNER_SYSTEM,
    ROBOT_PLANNER_USER,
    format_named_waypoints,
    format_skill_descriptions,
)
from core.shared_state import LLMOutput, SharedState

logger = logging.getLogger(__name__)


# =====================================================================
# Safety bounds — match config/settings.json defaults
# =====================================================================


class SafetyBounds:
    """Mutable container — the AI mode handler updates from settings on save."""

    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 1.5,
        min_battery_pct: float = 15.0,
        low_battery_factor: float = 0.5,
        cliff_escape_distance_m: float = 0.15,
    ) -> None:
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.min_battery_pct = min_battery_pct
        self.low_battery_factor = low_battery_factor
        self.cliff_escape_distance_m = cliff_escape_distance_m


# =====================================================================
# LLMPlanner
# =====================================================================


class LLMPlanner(QObject):
    """LLM-driven action selection with JSON validation and safety overrides."""

    plan_started = pyqtSignal(str)        # model name
    plan_finished = pyqtSignal(object)    # LLMOutput

    def __init__(
        self,
        state: Optional[SharedState] = None,
        *,
        openai_key: str = "",
        google_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        safety: Optional[SafetyBounds] = None,
        performance_monitor: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state or SharedState.instance()
        self.openai_key = openai_key
        self.google_key = google_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.safety = safety or SafetyBounds()
        self._perf = performance_monitor

    # ---------------------------------------------------------------
    # Public configuration
    # ---------------------------------------------------------------

    def set_api_keys(self, openai_key: str, google_key: str) -> None:
        self.openai_key = openai_key or ""
        self.google_key = google_key or ""

    def set_safety(self, safety: SafetyBounds) -> None:
        self.safety = safety

    # ---------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------

    async def plan_async(
        self,
        task_description: str,
        vlm_text: str,
        model_name: str,
        *,
        execution_type: str = "high_level",   # "high_level" | "low_level"
        allowed_skills: Optional[List[str]] = None,
        skill_descriptions: Optional[Dict[str, Any]] = None,
    ) -> LLMOutput:
        """Build context, call the LLM, parse, apply safety. Returns LLMOutput."""
        info = get_model(model_name)
        self.plan_started.emit(info.name)

        snap = self.state.snapshot_for_planner()
        yolo_detections_json = json.dumps(snap.get("detections", []), indent=2)
        # Feed a generous slice of history. Each iteration appends ~2 records
        # (the planning record + the post-execution skill_result), so a
        # 5-record window only covered ~2.5 iterations — far too short for a
        # multi-step mission (the planner would forget it had already reached
        # an earlier target and restart that sub-goal). 16 covers ~8
        # iterations, enough to see every completed sub-goal of a typical
        # find→approach→find→approach→return mission. The records are tiny.
        action_history_json = json.dumps(snap.get("action_history", [])[-16:], indent=2)

        prompt_user = ROBOT_PLANNER_USER.format(
            task_description=task_description or "(no task)",
            execution_mode=execution_type,
            allowed_skills=", ".join(allowed_skills or []) or "(none)",
            skill_descriptions=format_skill_descriptions(skill_descriptions or {}),
            vlm_text=vlm_text or "(no VLM analysis)",
            yolo_detections_json=yolo_detections_json,
            pose_x=snap["odom"]["x"],
            pose_y=snap["odom"]["y"],
            pose_yaw_deg=float(np.degrees(snap["odom"]["yaw_rad"])),
            battery_percent=snap["battery_percent"],
            bumpers_active=snap["bumpers_active"],
            cliff_active=snap["cliff_active"],
            wheel_drop_active=snap["wheel_drop_active"],
            trajectory_len=len(self.state.slam_trajectory),
            named_waypoints=format_named_waypoints(snap["named_waypoints"]),
            manual_waypoints_count=snap["manual_waypoints_count"],
            remembered_objects=snap.get("remembered_objects", "(none)"),
            action_history_json=action_history_json,
        )

        t0 = time.monotonic()
        try:
            raw_text, tokens = await self._dispatch(info, prompt_user)
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            output = self._safe_blocked_output(
                f"LLM call failed: {type(exc).__name__}: {exc}",
                model=info.name,
            )
            self.plan_finished.emit(output)
            return output

        latency_ms = (time.monotonic() - t0) * 1000.0
        if self._perf is not None:
            self._perf.record_llm_call(latency_ms, tokens=tokens)

        parsed = self._parse_json(raw_text)
        if parsed is None:
            output = self._safe_blocked_output(
                "Planner JSON parse failed — robot stopped.",
                model=info.name,
                raw_text=raw_text,
            )
        else:
            output = self._build_output(
                parsed, raw_text=raw_text, model=info.name,
                latency_ms=latency_ms, tokens=tokens,
            )

        # Safety overrides
        output = self._apply_safety(output)
        self.state.set_llm_output(output)
        action_record: Dict[str, Any] = {
            "model": output.model,
            "action_type": output.action_type,
            "status": output.status,
            "reasoning_excerpt": output.reasoning[:160],
        }
        if output.action_type == "skill" and isinstance(output.skill_command, dict):
            action_record["skill_name"] = output.skill_command.get("skill_name")
            action_record["params"] = output.skill_command.get("parameters") or {}
        elif output.action_type == "low_level" and isinstance(output.low_level_command, dict):
            action_record["low_level_command"] = output.low_level_command
        self.state.append_action(action_record)
        self.plan_finished.emit(output)
        return output

    # ---------------------------------------------------------------
    # Provider dispatch
    # ---------------------------------------------------------------

    async def _dispatch(self, info: ModelInfo, prompt_user: str
                        ) -> Tuple[str, int]:
        if info.provider == Provider.OPENAI:
            return await self._call_openai(info, prompt_user)
        if info.provider == Provider.GOOGLE:
            return await self._call_google(info, prompt_user)
        if info.provider == Provider.FUSION:
            return await self._call_fusion(info, prompt_user)
        if info.provider == Provider.LOCAL:
            raise RuntimeError(
                f"Model '{info.name}' has no LLM capability — choose a remote model."
            )
        raise ValueError(f"Unsupported provider: {info.provider}")

    async def _call_openai(self, info: ModelInfo, prompt_user: str
                           ) -> Tuple[str, int]:
        if not self.openai_key:
            raise RuntimeError("OpenAI API key not configured")
        import openai
        client = openai.AsyncOpenAI(api_key=self.openai_key)
        kwargs: Dict[str, Any] = {
            "model": info.api_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": ROBOT_PLANNER_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
        }
        if info.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    async def _call_google(self, info: ModelInfo, prompt_user: str
                           ) -> Tuple[str, int]:
        if not self.google_key:
            raise RuntimeError("Google API key not configured")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._call_google_sync, info, prompt_user,
        )

    def _call_google_sync(self, info: ModelInfo, prompt_user: str
                          ) -> Tuple[str, int]:
        import google.generativeai as genai
        genai.configure(api_key=self.google_key)
        cfg: Dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if info.supports_json_mode:
            cfg["response_mime_type"] = "application/json"
        model = genai.GenerativeModel(
            info.api_id,
            system_instruction=ROBOT_PLANNER_SYSTEM,
        )
        resp = model.generate_content(prompt_user, generation_config=cfg)
        try:
            text = resp.text or ""
        except Exception:
            text = str(resp)
        tokens = 0
        try:
            tokens = int(getattr(resp, "usage_metadata", None).total_token_count)  # type: ignore[union-attr]
        except Exception:
            pass
        return text, tokens

    async def _call_fusion(self, info: ModelInfo, prompt_user: str
                           ) -> Tuple[str, int]:
        gpt_info = MODELS["gpt-4o"]
        gem_info = MODELS["gemini-1.5-pro"]
        results = await asyncio.gather(
            self._call_openai(gpt_info, prompt_user),
            self._call_google(gem_info, prompt_user),
            return_exceptions=True,
        )
        parsed_outputs: List[Dict[str, Any]] = []
        tokens_total = 0
        raw_texts: List[str] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Fusion member failed: %s", r)
                continue
            text, t = r
            tokens_total += t
            raw_texts.append(text)
            j = self._parse_json(text)
            if j is not None:
                parsed_outputs.append(j)
        if not parsed_outputs:
            raise RuntimeError("Fusion: both LLM members failed or returned non-JSON")
        merged = _merge_planner_jsons(parsed_outputs)
        return json.dumps(merged), tokens_total

    # ---------------------------------------------------------------
    # JSON parsing + output construction
    # ---------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text.lstrip("`")
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to locate the outermost {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def _build_output(self, parsed: Dict[str, Any], *, raw_text: str,
                      model: str, latency_ms: float, tokens: int) -> LLMOutput:
        return LLMOutput(
            reasoning=str(parsed.get("reasoning", ""))[:2000],
            confidence=float(parsed.get("confidence", 0.0)),
            action_type=str(parsed.get("action_type", "")),
            low_level_command=parsed.get("low_level_command"),
            skill_command=parsed.get("skill_command"),
            status=str(parsed.get("status", "")),
            next_observation=str(parsed.get("next_observation", "")),
            raw_text=raw_text,
            model=model,
            latency_ms=latency_ms,
            tokens_used=tokens,
            monotonic_ts=time.monotonic(),
        )

    def _safe_blocked_output(self, reason: str, *, model: str,
                             raw_text: str = "") -> LLMOutput:
        """Synthesize a stop-action LLMOutput when the model fails / parsing breaks."""
        return LLMOutput(
            reasoning=reason,
            confidence=0.0,
            action_type="low_level",
            low_level_command={"linear_x": 0.0, "angular_z": 0.0, "duration_ms": 200},
            skill_command=None,
            status="blocked",
            next_observation="recover sensors and re-plan",
            raw_text=raw_text,
            model=model,
        )

    # ---------------------------------------------------------------
    # Safety layer
    # ---------------------------------------------------------------

    def _apply_safety(self, output: LLMOutput) -> LLMOutput:
        """Clamp velocities; intervene only when truly necessary.

        Philosophy: the safety layer used to override `status` to "blocked"
        on every bumper/cliff event, which made the robot bounce back into
        the LLM in a loop and feel timid. Now safety only PREVENTS DAMAGE
        (zero forward velocity on physical contact) and lets the LLM
        decide recovery — the LLM sees `bumpers_active=true` in its context
        and chooses whether to back away, turn, or genuinely give up.

        Annotations are appended to `reasoning` ONLY when something was
        actually changed, not on every clean pass.
        """
        with self.state.lock:
            bumpers_active = self.state.bumpers.any_active()
            cliff_active = self.state.cliff.any_active()
            wheel_drop_active = self.state.wheel_drop.any_active()
            battery_pct = self.state.battery.percent

        # Hard safety: physical contact → forbid forward motion this step.
        # We DO NOT change `status` to "blocked"; the LLM keeps task control.
        # We DO clamp linear_x to <= 0 (allow backing away) and zero angular
        # only if the LLM hasn't already chosen a sensible recovery action.
        if bumpers_active or wheel_drop_active:
            if output.action_type == "low_level" and isinstance(output.low_level_command, dict):
                cmd = dict(output.low_level_command)
                lin = float(cmd.get("linear_x", 0.0) or 0.0)
                # Permit backing up; forbid forward motion while contact is active.
                if lin > 0.0:
                    cmd["linear_x"] = 0.0
                    output.reasoning = (
                        output.reasoning
                        + " [safety: forward motion blocked by bumper; backing up or turning is allowed]"
                    ).strip()
                output.low_level_command = cmd
            elif output.action_type == "skill":
                # Force a brief stop before letting the skill run, but don't kill it
                output.low_level_command = {"linear_x": 0.0, "angular_z": 0.0, "duration_ms": 200}
                output.action_type = "low_level"
                output.reasoning = (
                    output.reasoning
                    + " [safety: bumper active — pausing skill briefly so the LLM can re-plan]"
                ).strip()

        # Cliff: stop forward motion only. No mandatory reverse — let the LLM
        # see `cliff_active=true` and decide whether to back up, turn, or stop.
        if cliff_active:
            if output.action_type == "low_level" and isinstance(output.low_level_command, dict):
                cmd = dict(output.low_level_command)
                lin = float(cmd.get("linear_x", 0.0) or 0.0)
                if lin > 0.0:
                    cmd["linear_x"] = 0.0
                    output.reasoning = (
                        output.reasoning
                        + " [safety: forward motion blocked by cliff sensor]"
                    ).strip()
                output.low_level_command = cmd

        # Velocity clamp (for low-level commands only); annotate only when
        # the value was actually changed.
        if output.action_type == "low_level" and isinstance(output.low_level_command, dict):
            cmd = dict(output.low_level_command)
            lin = float(cmd.get("linear_x", 0.0) or 0.0)
            ang = float(cmd.get("angular_z", 0.0) or 0.0)
            clamped_lin = _clamp(lin, -self.safety.max_linear, self.safety.max_linear)
            clamped_ang = _clamp(ang, -self.safety.max_angular, self.safety.max_angular)
            if clamped_lin != lin or clamped_ang != ang:
                output.reasoning = (
                    output.reasoning
                    + f" [safety: clamped to vmax={self.safety.max_linear:.2f},"
                      f" wmax={self.safety.max_angular:.2f}]"
                ).strip()
            cmd["linear_x"] = clamped_lin
            cmd["angular_z"] = clamped_ang
            try:
                duration_in = int(cmd.get("duration_ms", 500))
            except (TypeError, ValueError):
                duration_in = 500
            cmd["duration_ms"] = max(50, min(5000, duration_in))
            # Low battery scaling — silent unless we actually scaled.
            if (battery_pct > 0.0
                    and battery_pct < self.safety.min_battery_pct
                    and (cmd["linear_x"] != 0.0 or cmd["angular_z"] != 0.0)):
                cmd["linear_x"] *= self.safety.low_battery_factor
                cmd["angular_z"] *= self.safety.low_battery_factor
                output.reasoning = (
                    output.reasoning
                    + f" [safety: low battery {battery_pct:.0f}%, speed × {self.safety.low_battery_factor:.2f}]"
                ).strip()
            output.low_level_command = cmd

        return output


# =====================================================================
# Helpers
# =====================================================================


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _merge_planner_jsons(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fusion merge — majority vote on action_type, average on velocities."""
    if not outputs:
        return {}
    if len(outputs) == 1:
        return outputs[0]

    # Action type: majority vote
    types = [o.get("action_type", "") for o in outputs]
    chosen_type = max(set(types), key=types.count)

    # Status: majority too
    statuses = [o.get("status", "") for o in outputs]
    chosen_status = max(set(statuses), key=statuses.count)

    merged: Dict[str, Any] = {
        "reasoning": " || ".join(
            f"[m{i+1}] {o.get('reasoning', '')[:300]}" for i, o in enumerate(outputs)
        ),
        "confidence": sum(float(o.get("confidence", 0.0)) for o in outputs) / len(outputs),
        "action_type": chosen_type,
        "status": chosen_status,
        "next_observation": outputs[0].get("next_observation", ""),
    }

    if chosen_type == "low_level":
        cmds = [o.get("low_level_command") for o in outputs
                if isinstance(o.get("low_level_command"), dict)]
        if cmds:
            merged["low_level_command"] = {
                "linear_x": sum(float(c.get("linear_x", 0.0) or 0.0) for c in cmds) / len(cmds),
                "angular_z": sum(float(c.get("angular_z", 0.0) or 0.0) for c in cmds) / len(cmds),
                "duration_ms": int(sum(int(c.get("duration_ms", 500) or 500) for c in cmds) / len(cmds)),
            }
    elif chosen_type == "skill":
        # Pick the skill_command from whichever member voted skill (first)
        for o in outputs:
            if o.get("action_type") == "skill" and isinstance(o.get("skill_command"), dict):
                merged["skill_command"] = o["skill_command"]
                break

    return merged
