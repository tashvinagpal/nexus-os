"""Fakes for voice-output tests: a TTS provider and an audio sink. No speakers needed."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from nexus.ai.providers.base import Locality, ProviderInfo
from nexus.ai.voice.base import AudioClip, HealthReport, TTSCapabilities, TTSError
from nexus.core.resources import ModelSpec, Placement

RATE = 24000


@dataclass
class FakeVoice:
    """A synthesizing TTSProvider + ModelBackend: 10 ms of audio per character."""

    name: str = "kokoro"
    ram_mb: int = 400
    placement: Placement = Placement.CPU
    fail: Exception | None = None       # raised by synthesize
    fail_load: bool = False
    synth_seconds: float = 0.0          # pretend synthesis takes this long
    calls: list = field(default_factory=list)
    loads: int = 0
    unloads: int = 0
    loaded: bool = False

    def __post_init__(self) -> None:
        self.info = ProviderInfo(name=f"tts:{self.name}@{self.placement.value}", locality=Locality.LOCAL,
                                 model=self.name)
        self.spec = ModelSpec(self.info.name, "tts", self.placement, self.ram_mb, 0, 120.0)
        self.voices = {"en": "af_heart", "hi": "hf_alpha"}
        self.stops = 0

    def available(self) -> bool:
        return True

    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(languages=("en", "hi"), voices=("af_heart", "hf_alpha"), sample_rates=(RATE,))

    def health_check(self) -> HealthReport:
        return HealthReport(True, "fake")

    def load(self) -> None:
        if self.fail_load:
            raise TTSError("couldn't load (model file damaged)")
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.unloads += 1
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded

    def synthesize(self, text, *, language=None, voice=None, speed=None) -> AudioClip:
        self.calls.append((text, language, voice))
        if self.synth_seconds:
            time.sleep(self.synth_seconds)
        if self.fail is not None:
            raise self.fail
        return AudioClip(b"\x01\x00" * (RATE // 100) * len(text), RATE)

    def speak(self, text, *, language=None) -> None:
        raise AssertionError("the Speaker plays through its sink, never provider.speak()")

    def stop(self) -> None:
        self.stops += 1


class FakePlayback:
    def __init__(self, sink: "FakeSink", clip: AudioClip) -> None:
        self.sink = sink
        self.clip = clip
        self.started_at = time.monotonic()
        self.error: str | None = sink.error_during_play
        self._done = threading.Event()
        self.stopped = False
        if sink.instant:
            self._done.set()
        else:
            threading.Timer(sink.play_seconds, self._done.set).start()

    def wait(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        self.stopped = True
        self._done.set()


class FakeSink:
    """Records what would be played. `instant` playback, or `play_seconds` per clip."""

    def __init__(self, *, device: str | None = "Speakers (Fake USB Audio)", instant: bool = True,
                 play_seconds: float = 0.2) -> None:
        self.device = device
        self.instant = instant
        self.play_seconds = play_seconds
        self.played: list[AudioClip] = []
        self.playbacks: list[FakePlayback] = []
        self.stops = 0
        self.error_on_play: Exception | None = None
        self.error_during_play: str | None = None

    def ready(self):
        return (True, self.device) if self.device else (False, "no audio output device is connected")

    def play(self, clip: AudioClip) -> FakePlayback:
        if self.error_on_play is not None:
            raise self.error_on_play
        self.played.append(clip)
        playback = FakePlayback(self, clip)
        self.playbacks.append(playback)
        return playback

    def stop(self) -> None:
        self.stops += 1
        for playback in self.playbacks:
            playback.stop()
