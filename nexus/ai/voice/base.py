"""Speech interfaces. Speech-to-text (subsystem 5) is faster-whisper behind
STTProvider; TTS and the wake word arrive in later subsystems. Every provider
with a model registers it with the ResourceManager so it loads only while needed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from nexus.ai.providers.base import Locality, ProviderInfo


@dataclass(frozen=True)
class AudioClip:
    pcm16: bytes          # mono, signed 16-bit little-endian
    sample_rate: int = 16000

    @property
    def seconds(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None   # e.g. "hi", "en", "pa"
    confidence: float | None = None  # 0-1: how sure the model is of these words (exp of mean log-prob)
    language_probability: float | None = None
    no_speech_probability: float | None = None
    audio_seconds: float | None = None
    seconds: float | None = None  # time spent transcribing
    model: str = ""
    speech_seconds: float | None = None  # speech found in the audio by voice-activity detection


class VoiceInputError(Exception):
    """Voice input produced no usable text. str(error) is written for the user."""


class ListeningCancelled(VoiceInputError):
    """The user cancelled (Ctrl+C) while NEXUS was listening or transcribing."""


class NoSpeech(VoiceInputError):
    """Nothing was said (silence, or only background noise)."""


class Unintelligible(VoiceInputError):
    """Something was said, but not clearly enough to act on."""


class STTUnavailable(VoiceInputError):
    """No speech-to-text model can run right now (not installed, no memory, game mode, GPU failure)."""


class STTTimeout(VoiceInputError):
    """Transcription took too long and was stopped."""


class STTFailed(VoiceInputError):
    """The speech-to-text engine failed (it is restarted on the next request)."""


class STTProvider(Protocol):
    """Speech-to-text. Local, LAN or cloud (Locality), chosen per request like chat models.

    A cloud provider only ever runs if the user enables cloud AI and a network
    is up (see nexus/ai/providers/selection.py); none exists yet.
    """

    info: ProviderInfo

    def available(self) -> bool: ...

    def transcribe(self, audio: AudioClip, *, language_hint: str | None = None,
                   cancel: threading.Event | None = None) -> Transcript:
        """Raise a VoiceInputError subclass instead of returning empty or guessed text."""
        ...


class TTSError(Exception):
    """Speech output failed (engine missing, model not loaded, synthesis error). Never fatal."""


@dataclass(frozen=True)
class TTSCapabilities:
    languages: tuple[str, ...] = ()        # NEXUS language keys it can speak: "en", "hi", ...
    voices: tuple[str, ...] = ()
    sample_rates: tuple[int, ...] = ()
    supports_streaming: bool = False       # audio before the whole text is synthesized
    supports_ssml: bool = False


@dataclass(frozen=True)
class HealthReport:
    ok: bool
    detail: str = ""


class TTSProvider(Protocol):
    """Replaceable voice output (Kokoro today; Piper or a cloud voice could implement it too).

    A provider turns text into audio (`synthesize`); the Speaker (nexus/ai/voice/speaker.py)
    prepares the text, splits it into sentences and plays the audio, so no provider needs to
    know about audio devices. Providers with a model also implement ModelBackend
    (spec/load/unload/is_loaded) so the ResourceManager controls when it is in memory.
    A cloud provider would only ever run if the user enabled cloud AI - none exists.
    """

    info: ProviderInfo

    def available(self) -> bool: ...

    def capabilities(self) -> TTSCapabilities: ...

    def health_check(self) -> HealthReport: ...

    def synthesize(self, text: str, *, language: str | None = None, voice: str | None = None,
                   speed: float | None = None) -> AudioClip:
        """Audio for this text. Raises TTSError instead of returning silence."""
        ...

    def speak(self, text: str, *, language: str | None = None) -> None:
        """Synthesize and play through the default output, returning when finished (simple uses;
        NEXUS itself speaks through the Speaker)."""
        ...

    def stop(self) -> None:
        """Stop speaking immediately (the user interrupted)."""
        ...


class SilentTTS:
    """Text-only output: used when no voice is configured. Says nothing, honestly."""

    info = ProviderInfo(name="tts:silent", locality=Locality.LOCAL)

    def available(self) -> bool:
        return True

    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities()

    def health_check(self) -> HealthReport:
        return HealthReport(True, "text only (no voice configured)")

    def speak(self, text: str, *, language: str | None = None) -> None:
        pass

    def stop(self) -> None:
        pass
