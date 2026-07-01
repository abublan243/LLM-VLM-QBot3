"""
Model Registry — single source of truth for every supported LLM/VLM.

Used by:
    * GUI dropdowns (Mode 1 control panel)
    * VLMPipeline (which provider to dispatch to)
    * LLMPlanner    (same)
    * Settings dialog (API key validation)

Adding a model = add an entry to MODELS. Nothing else needs to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    FUSION = "fusion"            # composite: dispatch to both & merge
    LOCAL  = "local"             # YOLO-only / no API call


class Capability(str, Enum):
    LLM = "llm"
    VLM = "vlm"
    LLM_VLM = "llm+vlm"


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for one model."""
    name: str                    # display & registry key
    provider: Provider
    capability: Capability
    api_id: str = ""             # provider-specific identifier passed to the SDK
    context_tokens: int = 128_000
    supports_vision: bool = True
    supports_json_mode: bool = True
    notes: str = ""
    fusion_members: Tuple[str, ...] = field(default_factory=tuple)

    def has_llm(self) -> bool:
        return self.capability in (Capability.LLM, Capability.LLM_VLM)

    def has_vlm(self) -> bool:
        return self.capability in (Capability.VLM, Capability.LLM_VLM)


# =====================================================================
# Registry
# =====================================================================

MODELS: Dict[str, ModelInfo] = {
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=Provider.OPENAI,
        capability=Capability.LLM_VLM,
        api_id="gpt-4o",
        context_tokens=128_000,
        supports_vision=True,
        supports_json_mode=True,
        notes="OpenAI flagship multimodal — strong on planning + vision.",
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        capability=Capability.LLM_VLM,
        api_id="gpt-4o-mini",
        context_tokens=128_000,
        supports_vision=True,
        supports_json_mode=True,
        notes="Cheaper, faster — recommended default for the planner.",
    ),
    "gemini-1.5-pro": ModelInfo(
        name="gemini-1.5-pro",
        provider=Provider.GOOGLE,
        capability=Capability.LLM_VLM,
        api_id="gemini-1.5-pro",
        context_tokens=1_000_000,
        supports_vision=True,
        supports_json_mode=True,
        notes="Massive context — useful for long task histories.",
    ),
    "gemini-1.5-flash": ModelInfo(
        name="gemini-1.5-flash",
        provider=Provider.GOOGLE,
        capability=Capability.LLM_VLM,
        api_id="gemini-1.5-flash",
        context_tokens=1_000_000,
        supports_vision=True,
        supports_json_mode=True,
        notes="Low-latency Gemini variant.",
    ),
    "gemini-robotics-er-1.6": ModelInfo(
        name="gemini-robotics-er-1.6",
        provider=Provider.GOOGLE,
        capability=Capability.LLM_VLM,
        api_id="gemini-robotics-er-1.6-preview",
        context_tokens=1_000_000,
        supports_vision=True,
        supports_json_mode=True,
        notes=(
            "Google DeepMind's robotics-tuned VLM (preview, released April 2026). "
            "Best for pointing, counting, success detection, instrument/gauge reading, "
            "and embodied spatial reasoning. Recommended VLM for the read_gauge, "
            "search_object, and sequential_approach skills. The SDK is "
            "`google-generativeai` (legacy) for now; the official `google-genai` SDK "
            "unlocks code_execution + thinking_budget but is not yet wired up here."
        ),
    ),
    "gpt4o+gemini-fusion": ModelInfo(
        name="gpt4o+gemini-fusion",
        provider=Provider.FUSION,
        capability=Capability.LLM_VLM,
        supports_vision=True,
        supports_json_mode=True,
        notes="Run GPT-4o and Gemini-1.5-Pro in parallel; merge by majority + average.",
        fusion_members=("gpt-4o", "gemini-1.5-pro"),
    ),
    "yolo-only": ModelInfo(
        name="yolo-only",
        provider=Provider.LOCAL,
        capability=Capability.VLM,
        api_id="",
        context_tokens=0,
        supports_vision=True,
        supports_json_mode=False,
        notes="No remote VLM — scene summary built from YOLO detections + depth only.",
    ),
}


# =====================================================================
# Lookup helpers
# =====================================================================


def get_model(name: str) -> ModelInfo:
    """Return the ModelInfo for `name`. Raises KeyError if unknown."""
    if name not in MODELS:
        raise KeyError(f"Unknown model: {name}. Known: {sorted(MODELS)}")
    return MODELS[name]


def llm_models() -> List[ModelInfo]:
    return [m for m in MODELS.values() if m.has_llm()]


def vlm_models() -> List[ModelInfo]:
    return [m for m in MODELS.values() if m.has_vlm()]


def llm_model_names() -> List[str]:
    return [m.name for m in llm_models()]


def vlm_model_names() -> List[str]:
    return [m.name for m in vlm_models()]


# =====================================================================
# API key validation (lightweight reachability check)
# =====================================================================


def validate_openai_key(api_key: str, *, timeout_s: float = 5.0
                        ) -> Tuple[bool, str]:
    """Returns (ok, message). Performs the cheapest possible OpenAI call."""
    if not api_key:
        return False, "no key set"
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=timeout_s)
        # models.list is the conventional cheap reachability probe
        client.models.list()
        return True, "OpenAI key OK"
    except Exception as exc:
        return False, f"OpenAI: {type(exc).__name__}: {exc}"


def validate_google_key(api_key: str, *, timeout_s: float = 5.0
                        ) -> Tuple[bool, str]:
    """Returns (ok, message). Probes google-generativeai's list_models.

    `timeout_s` is kept for API symmetry with validate_openai_key; the legacy
    google-generativeai SDK doesn't expose a per-call timeout knob, so it is
    effectively informational here.
    """
    del timeout_s
    if not api_key:
        return False, "no key set"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # iterate just the first model to confirm auth works
        for _ in genai.list_models():
            break
        return True, "Google key OK"
    except Exception as exc:
        return False, f"Google: {type(exc).__name__}: {exc}"


def validate_keys(openai_key: str, google_key: str
                  ) -> Dict[str, Tuple[bool, str]]:
    """Convenience wrapper used by the settings dialog."""
    return {
        "openai": validate_openai_key(openai_key),
        "google": validate_google_key(google_key),
    }


def required_provider_keys(model: ModelInfo) -> List[Provider]:
    """Which provider keys must be set for this model to function."""
    if model.provider == Provider.LOCAL:
        return []
    if model.provider == Provider.FUSION:
        return [get_model(name).provider for name in model.fusion_members]
    return [model.provider]


def is_model_runnable(model_name: str, openai_key: str, google_key: str
                      ) -> Tuple[bool, str]:
    """Cheap predicate — true only if every provider this model needs has a key configured."""
    try:
        info = get_model(model_name)
    except KeyError as exc:
        return False, str(exc)
    needed = required_provider_keys(info)
    missing: List[str] = []
    if Provider.OPENAI in needed and not openai_key:
        missing.append("OpenAI")
    if Provider.GOOGLE in needed and not google_key:
        missing.append("Google")
    if missing:
        return False, f"Missing API key(s): {', '.join(missing)}"
    return True, "OK"
