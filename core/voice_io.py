"""
VoiceIO — speech-to-text input and text-to-speech output for the operator.

* STT: OpenAI Whisper (`whisper-1`) via the existing api_keys.openai
* TTS: OpenAI TTS (`tts-1`) via the same key

Mic capture and audio playback go through `sounddevice` (PortAudio).
If `sounddevice` or the `openai` SDK isn't installed, the class loads in a
disabled state — `can_record()` / `can_speak()` return False and the GUI
hides the mic button / speak toggle gracefully. The rest of the app
continues to work.

All blocking calls (Whisper API, TTS API, audio playback) are offloaded to
`asyncio.to_thread` so the qasync GUI loop never stalls.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
    _HAVE_SD = True
except Exception as exc:                                # pragma: no cover
    sd = None                                            # type: ignore
    _HAVE_SD = False
    logger.info("sounddevice not available — voice I/O disabled (%s)", exc)

try:
    from openai import OpenAI  # type: ignore
    _HAVE_OPENAI = True
except Exception:                                        # pragma: no cover
    OpenAI = None                                        # type: ignore
    _HAVE_OPENAI = False


# Whisper accepts arbitrary sample rates but standardising on 16 kHz mono
# keeps the wav blob small and matches the model's training distribution.
SAMPLE_RATE_HZ = 16000


class VoiceIO:
    """Push-to-talk Whisper STT + OpenAI TTS playback. Stateless beyond a
    long-lived OpenAI client and an active recording stream (if any).
    """

    def __init__(
        self,
        openai_key: str = "",
        *,
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "nova",
        language: Optional[str] = None,
        # VAD knobs for the toggle-mode UI (no-op for hold-to-talk callers).
        # silence_rms_threshold: int16 RMS below which a block counts as silent.
        # silence_timeout_s: trailing-silence duration before auto-stop.
        # min_speech_s: don't auto-stop until we've heard at least this much
        #   audio above threshold — gives the user a few seconds of "thinking
        #   silence" at the start of the recording.
        silence_rms_threshold: float = 500.0,
        silence_timeout_s: float = 1.5,
        min_speech_s: float = 0.4,
    ) -> None:
        self.openai_key = openai_key or ""
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.language = language
        self._client = self._build_client()

        self.silence_rms_threshold = float(silence_rms_threshold)
        self.silence_timeout_s = float(silence_timeout_s)
        self.min_speech_s = float(min_speech_s)

        self._stream: Optional[Any] = None
        self._capture_buffer: List[np.ndarray] = []
        self._capture_lock = threading.Lock()
        self._playing_lock = threading.Lock()

        # VAD state — written from the audio thread, read from the GUI thread
        # (atomic float / bool reads so no lock needed for the polled fields).
        self._recording_started_ts: float = 0.0
        self._last_loud_ts: float = 0.0
        self._heard_speech: bool = False
        self._speech_seconds: float = 0.0

    def _build_client(self) -> Any:
        if not (_HAVE_OPENAI and self.openai_key):
            return None
        try:
            return OpenAI(api_key=self.openai_key)
        except Exception as exc:                         # pragma: no cover
            logger.warning("Could not build OpenAI client: %s", exc)
            return None

    # ---------------------------------------------------------------
    # Capability checks (the GUI uses these to hide buttons gracefully)
    # ---------------------------------------------------------------

    def can_record(self) -> bool:
        return _HAVE_SD and self._client is not None

    def can_speak(self) -> bool:
        return _HAVE_SD and self._client is not None

    def status_text(self) -> str:
        """Short string for tooltips / status pills."""
        if not _HAVE_SD:
            return "voice disabled — install sounddevice"
        if self._client is None:
            return "voice disabled — set api_keys.openai in Settings"
        return "voice ready"

    # ---------------------------------------------------------------
    # Push-to-talk recording
    # ---------------------------------------------------------------

    def start_recording(self) -> bool:
        """Open the input stream. Returns False on failure."""
        if not _HAVE_SD:
            return False
        if self._stream is not None:
            return True       # already recording
        import time as _time
        with self._capture_lock:
            self._capture_buffer = []
        # Reset VAD state for the new recording
        now = _time.monotonic()
        self._recording_started_ts = now
        self._last_loud_ts = now
        self._heard_speech = False
        self._speech_seconds = 0.0
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE_HZ,
                channels=1,
                dtype="int16",
                callback=self._on_audio_block,
            )
            self._stream.start()
            logger.info("VoiceIO: recording started")
            return True
        except Exception as exc:
            logger.exception("start_recording failed: %s", exc)
            self._stream = None
            return False

    def _on_audio_block(self, indata, frames, time_info, status) -> None:  # noqa: D401, ARG002
        import time as _time
        # Cheap RMS on int16 samples — divides by sqrt(N) implicitly via mean.
        # Using float64 to avoid int16 overflow on sum-of-squares.
        try:
            arr = indata.reshape(-1).astype(np.float64)
            if arr.size > 0:
                rms = float(np.sqrt(np.mean(arr * arr)))
                block_seconds = arr.size / float(SAMPLE_RATE_HZ)
                if rms > self.silence_rms_threshold:
                    self._last_loud_ts = _time.monotonic()
                    self._heard_speech = True
                    self._speech_seconds += block_seconds
        except Exception:
            pass
        with self._capture_lock:
            self._capture_buffer.append(indata.copy())

    def should_auto_stop(self) -> bool:
        """True when toggle-mode UI should programmatically stop recording.

        Auto-stop fires only after we've heard at least `min_speech_s` of
        above-threshold audio AND the trailing silence has lasted
        `silence_timeout_s`. This lets the operator pause briefly mid-thought
        without the recording cutting off, but stops cleanly once they're done.
        """
        if self._stream is None:
            return False
        if not self._heard_speech:
            return False
        if self._speech_seconds < self.min_speech_s:
            return False
        import time as _time
        return (_time.monotonic() - self._last_loud_ts) > self.silence_timeout_s

    def stop_recording(self) -> Optional[bytes]:
        """Stop the input stream and return the captured audio as a wav blob.

        Returns None if no audio was captured (or recording wasn't active).
        """
        if self._stream is None:
            return None
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:                         # pragma: no cover
            logger.warning("stop_recording cleanup: %s", exc)
        finally:
            self._stream = None
        with self._capture_lock:
            chunks = self._capture_buffer
            self._capture_buffer = []
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0).astype(np.int16)
        # < ~0.4 s of audio is almost certainly an accidental click — Whisper
        # would just hallucinate.
        if audio.shape[0] < int(0.4 * SAMPLE_RATE_HZ):
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE_HZ)
            w.writeframes(audio.tobytes())
        return buf.getvalue()

    def is_recording(self) -> bool:
        return self._stream is not None

    # ---------------------------------------------------------------
    # STT
    # ---------------------------------------------------------------

    async def transcribe(self, wav_bytes: bytes) -> str:
        if self._client is None or not wav_bytes:
            return ""
        return await asyncio.to_thread(self._transcribe_blocking, wav_bytes)

    def _transcribe_blocking(self, wav_bytes: bytes) -> str:
        bio = io.BytesIO(wav_bytes)
        bio.name = "speech.wav"      # SDK uses the .name attr to set MIME
        try:
            kwargs: dict = {"model": self.stt_model, "file": bio}
            if self.language:
                kwargs["language"] = self.language
            resp = self._client.audio.transcriptions.create(**kwargs)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as exc:
            logger.exception("Whisper STT failed: %s", exc)
            return ""

    # ---------------------------------------------------------------
    # TTS
    # ---------------------------------------------------------------

    async def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or self._client is None or not _HAVE_SD:
            return False
        return await asyncio.to_thread(self._speak_blocking, text)

    def _speak_blocking(self, text: str) -> bool:
        try:
            with self._client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                response_format="wav",
            ) as resp:
                buf = io.BytesIO()
                for chunk in resp.iter_bytes(8192):
                    buf.write(chunk)
            buf.seek(0)
            with wave.open(buf, "rb") as w:
                rate = w.getframerate()
                channels = w.getnchannels()
                width = w.getsampwidth()
                frames = w.readframes(w.getnframes())
            dtype = np.int16 if width == 2 else np.int8
            arr = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                arr = arr.reshape(-1, channels)
            with self._playing_lock:
                sd.stop()              # interrupt any previous playback
                sd.play(arr, samplerate=rate, blocking=True)
            return True
        except Exception as exc:
            logger.exception("TTS speak failed: %s", exc)
            return False

    def stop_speaking(self) -> None:
        if _HAVE_SD:
            try:
                sd.stop()
            except Exception:                            # pragma: no cover
                pass

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def set_api_key(self, key: str) -> None:
        """Update the API key (e.g. after the user edits it in Settings)."""
        self.openai_key = key or ""
        self._client = self._build_client()

    def set_voice(self, voice: str) -> None:
        if voice:
            self.tts_voice = voice

    def set_language(self, language: Optional[str]) -> None:
        self.language = language or None

    def shutdown(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:                                # pragma: no cover
            pass
        self.stop_speaking()
