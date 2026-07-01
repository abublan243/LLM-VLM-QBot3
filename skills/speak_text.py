"""
Speak Text — text-to-speech skill using OpenAI TTS.

The LLM (or block-program) supplies a ``text`` string and the robot plays
it aloud through the host speaker using the same OpenAI TTS pipeline that
``VoiceIO`` already provides (model ``tts-1``, voice ``nova``).

No locomotion is involved — the robot stays still while speaking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from skills.base_skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class SpeakTextSkill(BaseSkill):
    name = "speak_text"
    description = "Speak the given text aloud using OpenAI TTS (text-to-speech)."
    icon = "speak"

    def __init__(
        self,
        state: Any,
        ros: Any,
        *,
        voice_io: Any = None,
        skills_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(state, ros, skills_config=skills_config)
        self._voice = voice_io

    async def _execute(self, params: Dict[str, Any]) -> SkillResult:
        text = str(params.get("text", "")).strip()
        if not text:
            return SkillResult(
                success=False,
                message="no text provided — set the 'text' parameter",
            )

        # Guard: VoiceIO must be available and configured
        if self._voice is None:
            return SkillResult(
                success=False,
                message="voice_io not available — check sounddevice + OpenAI key",
            )
        if not getattr(self._voice, "can_speak", lambda: False)():
            return SkillResult(
                success=False,
                message="TTS not ready — ensure sounddevice is installed and the OpenAI API key is set",
            )

        self._set(progress=0.1, status="synthesising speech")
        logger.info("SpeakText: speaking %d chars", len(text))

        try:
            ok = await self._voice.speak(text)
        except Exception as exc:
            logger.exception("SpeakText TTS failed: %s", exc)
            return SkillResult(success=False, message=f"TTS failed: {exc}")

        if ok:
            self._set(progress=1.0, status="done")
            msg = f"spoke: {text[:80]}{'…' if len(text) > 80 else ''}"
            self.state.append_event("INFO", msg)
            return SkillResult(success=True, message=msg)
        else:
            return SkillResult(success=False, message="TTS returned False — playback may have failed")
