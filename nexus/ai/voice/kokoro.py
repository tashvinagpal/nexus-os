"""Kokoro-82M text-to-speech: local, offline, loaded only while needed.

`KokoroTTS` is a TTSProvider and a ModelBackend: the ResourceManager decides
when it loads (lazily, on the first thing to say) and unloads it when idle.
"Loaded" means a worker process (kokoro_worker.py) holds the ONNX model;
unloading ends that process, which returns every byte it used.

Model files (Apache-2.0) live in models/tts/kokoro-v1.0/ and are never
downloaded automatically: scripts/get_tts_model.py shows the size first.
Nothing is sent anywhere. English and Hindi voices are validated on this PC
(docs/tts.md); the model has voices for a few other languages that NEXUS
doesn't use yet.
"""

from __future__ import annotations

import logging
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from nexus.ai.providers.base import Locality, ProviderInfo
from nexus.ai.voice.base import AudioClip, HealthReport, TTSCapabilities, TTSError
from nexus.ai.voice.worker_process import WorkerError, WorkerProcess, windowless_python
from nexus.core.config import PROJECT_ROOT
from nexus.core.resources import ModelSpec, Placement

log = logging.getLogger(__name__)

# variant -> (file name, download size in MB)
KOKORO_FILES = {
    "int8": ("kokoro-v1.0.int8.onnx", 88),
    "fp16": ("kokoro-v1.0.fp16.onnx", 169),
    "fp32": ("kokoro-v1.0.onnx", 310),
    "voices": ("voices-v1.0.bin", 26),
}
MODELS_DIR = PROJECT_ROOT / "models" / "tts" / "kokoro-v1.0"
SAMPLE_RATE = 24000
# NEXUS language key -> espeak-ng language used to turn text into phonemes. Validated here: en, hi.
LANGUAGES = {"en": "en-us", "hi": "hi"}
_BROKEN_FOR_S = 300.0


class KokoroTTS:
    def __init__(self, variant: str = "int8", *, voices: dict[str, str] | None = None, speed: float = 1.0,
                 volume: float = 1.0, threads: int = 4, placement: Placement = Placement.CPU, ram_mb: int = 400,
                 idle_unload_s: float = 120.0, timeout_s: float = 30.0, load_timeout_s: float = 60.0,
                 models_dir: Path = MODELS_DIR, log_file: Path | None = None, command: list[str] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if variant not in KOKORO_FILES or variant == "voices":
            raise ValueError(f"unknown Kokoro model variant {variant!r}")
        self.variant = variant
        self.model_file = models_dir / KOKORO_FILES[variant][0]
        self.voices_file = models_dir / KOKORO_FILES["voices"][0]
        self.voices = {"en": "af_heart", "hi": "hf_alpha", **(voices or {})}
        self.speed = speed
        self.volume = volume
        self.threads = threads
        self.placement = placement
        self.timeout_s = timeout_s
        self.load_timeout_s = load_timeout_s
        self._clock = clock
        self.info = ProviderInfo(name=f"tts:kokoro@{placement.value}", locality=Locality.LOCAL,
                                 model=f"kokoro-v1.0-{variant}")
        self.spec = ModelSpec(self.info.name, "tts", placement, ram_mb, 0, idle_unload_s)
        cmd = command or [windowless_python(), "-m", "nexus.ai.voice.kokoro_worker", "--model", str(self.model_file),
                          "--voices", str(self.voices_file), "--threads", str(threads),
                          "--device", "gpu" if placement is Placement.GPU else "cpu"]
        self._fake = command is not None  # tests run a fake worker
        self._worker = WorkerProcess(cmd, cwd=PROJECT_ROOT, log_file=log_file,
                                     env={"OMP_WAIT_POLICY": "PASSIVE", "PYTHONIOENCODING": "utf-8"}, clock=clock)
        self._broken_until = 0.0
        self._own_sink = None  # only for speak(); NEXUS plays through the Speaker's sink
        self.problem = ""
        self.last_load_seconds: float | None = None
        self.last_synthesis: dict = {}

    # --- availability -----------------------------------------------------------------

    def available(self) -> bool:
        if self._clock() < self._broken_until:
            return False
        if self._fake:
            return True
        if find_spec("kokoro_onnx") is None:
            self.problem = "kokoro-onnx isn't installed (pip install -e \".[tts]\")"
            return False
        missing = [f.name for f in (self.model_file, self.voices_file) if not f.is_file()]
        if missing:
            self.problem = (f"model not downloaded ({', '.join(missing)}) - run: .venv\\Scripts\\python.exe "
                            f"scripts\\get_tts_model.py {self.variant}")
            return False
        self.problem = ""
        return True

    def capabilities(self) -> TTSCapabilities:
        voices = tuple(self._worker.ready.get("voices", ())) or tuple(sorted(set(self.voices.values())))
        return TTSCapabilities(languages=tuple(LANGUAGES), voices=voices, sample_rates=(SAMPLE_RATE,),
                               supports_streaming=False, supports_ssml=False)

    def health_check(self) -> HealthReport:
        if not self.available():
            return HealthReport(False, self.problem or "not available")
        if not self.is_loaded():
            return HealthReport(True, "installed; loads on first use")
        try:
            self._worker.request({"op": "ping"}, timeout_s=5)
            return HealthReport(True, f"loaded (pid {self._worker.pid}, {self._worker.memory_mb()} MB)")
        except WorkerError as exc:
            return HealthReport(False, f"the speech engine didn't answer: {exc}")

    # --- ModelBackend --------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._worker.alive

    @property
    def pid(self) -> int | None:
        return self._worker.pid

    def memory_mb(self) -> int:
        return self._worker.memory_mb()

    def load(self) -> None:
        if self.is_loaded():
            return
        if not self.available():
            raise TTSError(f"{self.info.name}: {self.problem or 'not available'}")
        started = time.perf_counter()
        try:
            ready = self._worker.start(self.load_timeout_s)
        except WorkerError as exc:
            self._broken_until = self._clock() + _BROKEN_FOR_S
            self.problem = f"couldn't load {self.info.name}: {exc}"
            log.warning("%s", self.problem)
            raise TTSError(self.problem) from None
        self.last_load_seconds = time.perf_counter() - started
        if ready.get("note"):
            log.info("%s: %s", self.info.name, ready["note"])
        log.info("%s loaded in %.1f s (pid %s, providers %s)", self.info.name, self.last_load_seconds,
                 self._worker.pid, ready.get("providers"))

    def unload(self) -> None:
        if self._worker.alive:
            self._worker.stop()
            log.info("%s unloaded", self.info.name)

    # --- TTSProvider ------------------------------------------------------------------------

    def synthesize(self, text: str, *, language: str | None = None, voice: str | None = None,
                   speed: float | None = None) -> AudioClip:
        text = (text or "").strip()
        if not text:
            raise TTSError("There's nothing to say.")
        lang = language if language in LANGUAGES else "en"
        voice = voice or self.voices.get(lang) or self.voices["en"]
        if not self.is_loaded():
            raise TTSError("The speech engine isn't running.")
        try:
            reply, pcm = self._worker.request(
                {"op": "synthesize", "text": text, "voice": voice, "lang": LANGUAGES[lang],
                 "speed": speed or self.speed, "volume": self.volume}, timeout_s=self.timeout_s)
        except WorkerError as exc:
            raise TTSError(f"Speech synthesis failed: {exc} (the engine restarts next time)") from None
        if not reply.get("ok"):
            raise TTSError(f"Speech synthesis failed: {reply.get('error')}")
        self.last_synthesis = {"seconds": reply.get("seconds"), "audio_seconds": reply.get("audio_seconds"),
                               "characters": len(text), "language": lang, "voice": voice}
        return AudioClip(pcm, int(reply.get("rate", SAMPLE_RATE)))

    def speak(self, text: str, *, language: str | None = None) -> None:
        """Synthesize and play on the default output (NEXUS itself speaks through the Speaker)."""
        from nexus.system.audio.playback import PortAudioSink

        self.load()
        self._own_sink = self._own_sink or PortAudioSink()
        self._own_sink.play(self.synthesize(text, language=language)).wait()

    def stop(self) -> None:
        """Stop speak()'s playback. A synthesis in progress (one sentence, under a second) is left to
        finish - ending the engine would force a reload - and the Speaker discards its audio."""
        if self._own_sink is not None:
            self._own_sink.stop()
