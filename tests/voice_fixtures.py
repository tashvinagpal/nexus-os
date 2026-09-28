"""Fakes for voice-input tests: microphones, audio streams, speech-to-text. No hardware needed."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from nexus.ai.providers.base import Locality, ProviderInfo
from nexus.ai.voice.base import AudioClip, STTUnavailable, Transcript
from nexus.core.resources import ModelSpec, Placement
from nexus.system.audio.errors import MicrophoneDenied, MicrophoneUnavailable
from nexus.system.platform.base import AudioInput

RATE = 16000


def mic(name: str, *, id: str | None = None, state: str = "active", form_factor: str = "microphone",
        bus: str = "usb", loopback: bool = False, remote: bool = False, default: bool = False,
        comms: bool = False) -> AudioInput:
    return AudioInput(id=id or f"id:{name}", name=name, state=state, form_factor=form_factor, bus=bus,
                      loopback=loopback, remote=remote, default=default, default_communications=comms)


class FakeDevices:
    """The OS device list; tests plug and unplug devices by editing `.inputs`."""

    def __init__(self, *inputs: AudioInput) -> None:
        self.inputs = list(inputs)
        self.calls = 0
        self.error: Exception | None = None

    def __call__(self) -> list[AudioInput]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.inputs)

    def plug(self, dev: AudioInput) -> None:
        self.inputs.append(dev)

    def unplug(self, name: str) -> None:
        self.inputs = [d for d in self.inputs if d.name != name]


# --- synthetic audio (deterministic, no numpy) ----------------------------------------

def _pack(samples) -> bytes:
    from array import array

    return array("h", (max(-32768, min(32767, int(s))) for s in samples)).tobytes()


def silence(ms: int, rate: int = RATE) -> bytes:
    return b"\x00\x00" * (rate * ms // 1000)


def noise(ms: int, amplitude: float = 100.0, rate: int = RATE, seed: int = 1) -> bytes:
    state, out = seed, []
    for _ in range(rate * ms // 1000):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append((state / 0x7FFFFFFF * 2 - 1) * amplitude * 1.7)
    return _pack(out)


def voice(ms: int, amplitude: float = 4000.0, rate: int = RATE) -> bytes:
    """A speech-like signal: a 180 Hz tone with a 4 Hz syllable rhythm (never fully silent)."""
    n = rate * ms // 1000
    return _pack(amplitude * (0.6 + 0.4 * math.sin(2 * math.pi * 4 * i / rate)) * math.sin(2 * math.pi * 180 * i / rate)
                 for i in range(n))


def mix(a: bytes, b: bytes) -> bytes:
    from array import array

    x, y = array("h"), array("h")
    x.frombytes(a)
    y.frombytes(b)
    return _pack(x[i] + (y[i] if i < len(y) else 0) for i in range(len(x)))


# --- capture ----------------------------------------------------------------------------

class FakeStream:
    """Delivers `audio` in 30 ms blocks; then either ends (stalls) or keeps sending `tail`."""

    def __init__(self, audio: bytes, *, rate: int = RATE, block_ms: int = 30, tail: bytes | None = None,
                 realtime: bool = False) -> None:
        self.sample_rate = rate
        self._block = rate * block_ms // 1000 * 2
        self._block_s = block_ms / 1000
        self._audio = audio
        self._pos = 0
        self._tail = tail
        self._tail_pos = 0
        self.realtime = realtime  # pace reads like a real microphone
        self.closed = False
        self.reads = 0

    def read(self, timeout: float) -> bytes | None:
        self.reads += 1
        if self.realtime:
            time.sleep(self._block_s)
        if self._pos < len(self._audio):
            block = self._audio[self._pos:self._pos + self._block]
            self._pos += self._block
            return block
        if self._tail:  # repeats forever
            block = self._tail[self._tail_pos:self._tail_pos + self._block]
            self._tail_pos = (self._tail_pos + self._block) % len(self._tail)
            return block
        return None  # the device stopped sending

    @property
    def active(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


class FakeCapture:
    def __init__(self, streams: dict[str, FakeStream] | None = None) -> None:
        self.streams = streams or {}
        self.refused: dict[str, Exception] = {}
        self.opened: list[str] = []
        self.refreshes = 0

    def open(self, mic: AudioInput, *, sample_rate: int = RATE, block_ms: int = 30):
        self.opened.append(mic.name)
        if mic.name in self.refused:
            raise self.refused[mic.name]
        if mic.name not in self.streams:
            raise MicrophoneUnavailable(f"{mic.name} isn't available to open right now.")
        return self.streams[mic.name]

    def refresh(self) -> None:
        self.refreshes += 1

    def deny(self, name: str) -> None:
        self.refused[name] = MicrophoneDenied("Microphone access is blocked.")


class TickClock:
    """A clock that advances a little every time it is read (device-timeout tests)."""

    def __init__(self, step: float = 0.1) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


# --- speech to text -------------------------------------------------------------------------

@dataclass
class FakeSTT:
    """An STTProvider + ModelBackend with scripted results."""

    name: str = "fake"
    placement: Placement = Placement.CPU
    ram_mb: int = 300
    vram_mb: int = 0
    locality: Locality = Locality.LOCAL
    results: list = field(default_factory=list)  # Transcript or Exception, used in order (last repeats)
    is_available: bool = True
    fail_load: bool = False
    loads: int = 0
    unloads: int = 0
    calls: list = field(default_factory=list)
    loaded: bool = False

    def __post_init__(self) -> None:
        self.info = ProviderInfo(name=f"stt:{self.name}@{self.placement.value}", locality=self.locality,
                                 model=self.name)
        self.spec = (ModelSpec(self.info.name, "stt", self.placement, self.ram_mb, self.vram_mb, 60.0)
                     if self.locality is Locality.LOCAL else None)
        self.problem = ""
        self.load_started = threading.Event()

    def available(self) -> bool:
        return self.is_available

    def load(self) -> None:
        self.load_started.set()
        if self.fail_load:
            self.is_available = False
            raise STTUnavailable(f"{self.info.name}: couldn't load (CUDA libraries missing)")
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.unloads += 1
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded

    def transcribe(self, audio: AudioClip, *, language_hint=None, cancel=None) -> Transcript:
        self.calls.append((audio, language_hint))
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)] if self.results else \
            Transcript("hello", "en", 0.9)
        if isinstance(result, Exception):
            raise result
        return result


# --- synthetic speech from Windows' own offline voices (no speakers, no loopback) -----------

def synthesize(phrases: dict[str, str], folder, voices=("Microsoft David Desktop", "Microsoft Zira Desktop")):
    """Write 16 kHz mono WAVs with System.Speech; returns {(voice, name): path}. Windows only."""
    import subprocess
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Add-Type -AssemblyName System.Speech",
             "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
             "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer"]
    out = {}
    for voice_name in voices:
        lines.append(f"$s.SelectVoice('{voice_name}')")
        short = voice_name.split()[1]
        for name, text in phrases.items():
            path = folder / f"{short}_{name}.wav"
            safe = text.replace("'", "''")
            lines.append(f"$s.SetOutputToWaveFile('{path}', $fmt); $s.Speak('{safe}')")
            out[(short, name)] = path
    lines.append("$s.SetOutputToNull(); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "; ".join(lines)], check=True,
                   capture_output=True, timeout=120)
    return out


def read_wav(path) -> AudioClip:
    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return AudioClip(w.readframes(w.getnframes()), w.getframerate())
