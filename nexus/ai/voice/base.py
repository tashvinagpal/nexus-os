"""Speech interfaces. Implementations (faster-whisper STT, Kokoro/Piper TTS,
openWakeWord) arrive in v0.1 subsystems 5-7; each registers its model with
the ResourceManager so it loads only while needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nexus.ai.providers.base import Locality, ProviderInfo


@dataclass(frozen=True)
class AudioClip:
    pcm16: bytes          # mono, signed 16-bit little-endian
    sample_rate: int = 16000


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None   # e.g. "hi", "en", "pa"
    confidence: float | None = None


class STTProvider(Protocol):
    info: ProviderInfo

    def transcribe(self, audio: AudioClip, *, language_hint: str | None = None) -> Transcript: ...


class TTSProvider(Protocol):
    """Replaceable voice output. Providers with a model also implement ModelBackend
    (spec/load/unload/is_loaded) so the ResourceManager controls when it is in memory."""

    info: ProviderInfo

    def speak(self, text: str, *, language: str | None = None) -> None:
        """Speak and return when finished. stop() from another thread interrupts."""
        ...

    def stop(self) -> None:
        """Stop speaking immediately (the user interrupted)."""
        ...


class SilentTTS:
    """Text-only output: used when no voice is configured. Says nothing, honestly."""

    info = ProviderInfo(name="tts:silent", locality=Locality.LOCAL)

    def speak(self, text: str, *, language: str | None = None) -> None:
        pass

    def stop(self) -> None:
        pass
