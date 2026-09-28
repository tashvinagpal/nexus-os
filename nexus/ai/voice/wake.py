"""Wake word: "Nexus" starts listening, without a key press.

The microphone is read continuously, but nothing leaves the PC and nothing is
kept: audio is examined in memory, a few seconds at most, and dropped. Only
after the wake word does the normal voice pipeline run (listener.py) - the
same speech-to-text, fast path, planner, Action Kernel, policy, confirmations
and audit log as push-to-talk. The wake word grants nothing by itself.

Two detectors implement WakeDetector (chosen in config/nexus.toml [wake]):

  openwakeword  a small neural network trained for "Nexus" (oww.py): ~2% of one
                CPU core, runs on every 80 ms of audio, reacts in ~0.2 s.
                Needs models/wake/nexus.onnx (scripts/train_wake_model.py).
  whisper       works today with no training: the energy detector from
                recorder.py finds bursts of speech; the first ~1.6 s of a burst
                (never more, and at most N times a minute) goes to a light local
                Whisper model, and a burst that starts with "Nexus" wakes NEXUS.
                Costs a short transcription per burst of speech in the room.

    microphone -> WakeDetector.process(frame) -> WakeHit
               -> Recorder continues the same utterance ("Nexus, GTA chala de")
               -> speech-to-text -> "GTA chala de" -> conversation engine
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from nexus.ai.providers.base import Locality, ProviderInfo
from nexus.ai.voice.base import AudioClip, VoiceInputError
from nexus.ai.voice.recorder import EnergyVAD, VADConfig, level
from nexus.core.resources import ModelSpec, Placement, ResourceManager, ResourceUnavailable

log = logging.getLogger(__name__)

RATE = 16000


class WakeUnavailable(Exception):
    """This wake-word detector can't run (model missing, engine failed). str(error) is for the user."""


@dataclass(frozen=True)
class WakeHit:
    detector: str
    score: float | None = None      # openwakeword: the network's score; whisper: the transcript's confidence
    heard: str = ""                 # whisper: what the detector's model wrote
    lead_in: bytes = b""            # the utterance so far, wake word included (16 kHz PCM16)
    silence_ms: int = 0             # quiet after the last speech in lead_in
    alone: bool = False             # "Nexus" was said on its own and followed by a pause
    noise_floor: float | None = None


class WakeDetector(Protocol):
    info: ProviderInfo
    frame_ms: int                   # process() takes frames of exactly this length
    problem: str

    def available(self) -> bool: ...

    def start(self) -> None:
        """Load what's needed before listening. Raises WakeUnavailable."""
        ...

    def stop(self) -> None: ...

    def reset(self) -> None:
        """Forget the audio heard so far (after a hit, a pause, or NEXUS speaking)."""
        ...

    def process(self, frame: bytes) -> WakeHit | None: ...


# --- is this text addressed to NEXUS? ----------------------------------------------------------

# How speech-to-text writes "Nexus" (English models, and Hindi/Punjabi ones in their own scripts).
_FORMS = {"nexus", "nexas", "nexis", "nexes", "nexos", "nexxus", "neksus", "nexsus", "nexuss", "naxus",
          "नेक्सस", "नेक्सेस", "नैक्सस", "नेकसस", "नक्सस", "नेक्सस्", "नैक्सेस", "ਨੈਕਸਸ", "ਨੇਕਸਸ", "ਨੈਕਸੱਸ"}
_SPLIT_FORMS = {("next", "us"), ("necks", "us"), ("nex", "us"), ("neck", "sus"), ("neck", "says")}
_FILLERS = {"hey", "hi", "hello", "ok", "okay", "oye", "oi", "arre", "arey", "are", "yo", "ae", "ay", "haan", "ha",
            "suno", "hmm", "um", "uh", "अरे", "ओए", "हे", "हाय", "ओके", "सुनो"}
_PUNCT = "\"'.,!?;:-()[]“”‘’…।॥"
_TOKEN = re.compile(r"\S+")


def _near_nexus(word: str) -> bool:
    """One letter off "nexus" and starting like it ("nexux", "nexuz") - not "next" or "necks"."""
    if not word.startswith(("ne", "na")) or not 4 <= len(word) <= 7:
        return False
    a, b = word, "nexus"
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 1


def find_wake(text: str, *, skip: int = 0) -> tuple[bool, str]:
    """(True, the rest) if `text` begins with the wake word (after at most two fillers like "hey").

    `skip` words of anything may come first: audio kept from before an openWakeWord hit can start
    with the end of someone else's sentence."""
    tokens = [(m.start(), m.end(), m.group().strip(_PUNCT).lower()) for m in _TOKEN.finditer(text)]
    tokens = [t for t in tokens if t[2]]
    for i, (_, end, word) in enumerate(tokens[:3 + skip]):
        if word in _FORMS or _near_nexus(word):
            return True, text[end:].lstrip(" " + _PUNCT).strip()
        if i + 1 < len(tokens) and (word, tokens[i + 1][2]) in _SPLIT_FORMS:
            return True, text[tokens[i + 1][1]:].lstrip(" " + _PUNCT).strip()
        if word not in _FILLERS and i >= skip:
            break
    return False, text.strip()


def strip_wake(text: str, *, skip: int = 0) -> str:
    found, rest = find_wake(text, skip=skip)
    return rest if found else text.strip()


# --- detector 1: a light Whisper model on the start of each burst of speech ----------------------

class WhisperWakeDetector:
    """Whisper as a wake-word detector: no training, but it transcribes the start of every burst of speech.

    `stt` is a WhisperSTT (or anything with transcribe() and a ModelSpec); the ResourceManager loads it
    for each check and unloads it when idle, like any other model.
    """

    frame_ms = 30

    def __init__(self, stt: Any, resources: ResourceManager, *, check_after_s: float = 1.6,
                 burst_end_ms: int = 450, max_burst_s: float = 6.0, max_checks_per_minute: int = 20,
                 min_confidence: float = 0.3, language: str | None = "en", prompt: str | None = None,
                 vad: VADConfig | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.stt = stt
        self.resources = resources
        self.check_after_ms = int(check_after_s * 1000)
        self.burst_end_ms = burst_end_ms
        self.max_burst_ms = int(max_burst_s * 1000)
        self.max_checks_per_minute = max_checks_per_minute
        self.min_confidence = min_confidence
        self.language = language
        self.prompt = prompt
        self.cfg = vad or VADConfig()
        self._clock = clock
        self.info = ProviderInfo(name=f"wake:whisper-{getattr(stt.info, 'model', '') or stt.info.name}",
                                 locality=Locality.LOCAL, model=getattr(stt.info, "model", ""))
        self.problem = ""
        self.checks = 0
        self.skipped = 0
        self._recent_checks: deque[float] = deque()
        self._vad = EnergyVAD(self.cfg)
        self.reset()

    @property
    def model_name(self) -> str | None:
        spec = getattr(self.stt, "spec", None)
        return spec.name if spec is not None else None

    def available(self) -> bool:
        ok = self.stt.available()
        self.problem = "" if ok else (getattr(self.stt, "problem", "") or "the speech model isn't available")
        return ok

    def start(self) -> None:
        if not self.available():
            raise WakeUnavailable(self.problem)
        name = self.model_name
        if name is None:
            return
        try:  # load it now, so the first "Nexus" doesn't wait for it
            with self.resources.use(name):
                pass
        except (ResourceUnavailable, VoiceInputError) as exc:
            raise WakeUnavailable(str(exc)) from None

    def stop(self) -> None:
        pass

    def reset(self) -> None:
        frames = max(1, (self.cfg.pre_roll_ms + self.cfg.calibration_ms) // self.frame_ms)
        self._before: deque[bytes] = deque(maxlen=frames)
        self._recent: deque[bool] = deque(maxlen=max(1, self.cfg.start_window_ms // self.frame_ms))
        self._burst: list[bytes] = []
        self._in_burst = False
        self._checked = False
        self._voiced_ms = self._silence_ms = self._burst_ms = 0

    @property
    def noise_floor(self) -> float | None:
        return self._vad.noise

    def process(self, frame: bytes) -> WakeHit | None:
        cfg = self.cfg
        rms = level(frame)
        if not self._in_burst:
            self._before.append(frame)
            self._recent.append(self._vad.voiced(rms, speaking=False))
            if sum(self._recent) * cfg.frame_ms >= cfg.start_ms:
                self._in_burst, self._checked = True, False
                self._burst = list(self._before)
                self._voiced_ms, self._silence_ms, self._burst_ms = sum(self._recent) * cfg.frame_ms, 0, 0
            return None
        self._burst.append(frame)
        self._burst_ms += cfg.frame_ms
        if self._vad.voiced(rms, speaking=True):
            self._voiced_ms += cfg.frame_ms
            self._silence_ms = 0
        else:
            self._silence_ms += cfg.frame_ms
        ended = self._silence_ms >= self.burst_end_ms
        hit = None
        if not self._checked and (ended or self._burst_ms >= self.check_after_ms):
            self._checked = True
            if self._voiced_ms >= cfg.min_speech_ms:  # a click or a cough isn't worth a check
                hit = self._check(ended)
        if ended or self._burst_ms >= self.max_burst_ms:
            # A long burst (a TV, a conversation) is split, so "Nexus" said during it still gets a check.
            self._end_burst()
        return hit

    def _end_burst(self) -> None:
        self._in_burst = False
        self._burst = []
        self._before.clear()
        self._recent.clear()

    def _check(self, ended: bool) -> WakeHit | None:
        now = self._clock()
        while self._recent_checks and now - self._recent_checks[0] > 60:
            self._recent_checks.popleft()
        if len(self._recent_checks) >= self.max_checks_per_minute:
            self.skipped += 1
            if self.skipped == 1 or self.skipped % 50 == 0:
                log.info("wake word: %d bursts of speech skipped (limit of %d checks a minute)",
                         self.skipped, self.max_checks_per_minute)
            return None
        self._recent_checks.append(now)
        self.checks += 1
        clip = AudioClip(b"".join(self._burst), RATE)
        name = self.model_name
        try:
            with ExitStack() as hold:
                if name is not None:
                    hold.enter_context(self.resources.use(name))
                kwargs = {"prompt": self.prompt} if self.prompt else {}
                transcript = self.stt.transcribe(clip, language_hint=self.language, **kwargs)
        except (ResourceUnavailable, VoiceInputError) as exc:
            self.problem = str(exc)
            log.info("wake word check failed: %s", exc)
            return None
        self.problem = ""
        found, rest = find_wake(transcript.text)
        confidence = transcript.confidence
        log.debug("wake check: %r (confidence %s) -> %s", transcript.text, confidence, found)
        if not found or (confidence is not None and confidence < self.min_confidence):
            return None
        return WakeHit(self.info.name, confidence, transcript.text, clip.pcm16, self._silence_ms,
                       alone=ended and not rest, noise_floor=self._vad.noise)


# --- detector 2: openWakeWord ------------------------------------------------------------------

class OpenWakeWordDetector:
    """A trained "Nexus" network (openWakeWord format). Also a ModelBackend: the ResourceManager counts
    its memory, and the listener holds it loaded while it listens."""

    frame_ms = 80

    def __init__(self, model: Path, melspec: Path, embedding: Path, resources: ResourceManager | None = None, *,
                 threshold: float = 0.5, patience: int = 1, threads: int = 1, ram_mb: int = 60,
                 lead_in_ms: int = 1000, vad: VADConfig | None = None) -> None:
        self.paths = (model, melspec, embedding)
        self.resources = resources
        self.threshold = threshold
        self.patience = max(1, patience)
        self.threads = threads
        self.info = ProviderInfo(name=f"wake:openwakeword-{model.stem}", locality=Locality.LOCAL, model=model.stem)
        self.spec = ModelSpec(self.info.name, "wakeword", Placement.CPU, ram_mb, 0, math.inf)
        self.cfg = vad or VADConfig(frame_ms=80)
        self._lead_chunks = max(1, lead_in_ms // self.frame_ms)
        self._features: Any = None
        self._model: Any = None
        self._np: Any = None
        self._hold: ExitStack | None = None
        self.problem = ""
        self.last_score = 0.0
        self.peak_score = 0.0            # highest score since the last reset (for /wake test)
        self._registered = False
        self.reset()

    # --- availability / ModelBackend -------------------------------------------------------------

    def available(self) -> bool:
        from importlib.util import find_spec

        if find_spec("onnxruntime") is None or find_spec("numpy") is None:
            self.problem = "onnxruntime isn't installed (pip install -e \".[voice]\")"
            return False
        missing = [p.name for p in self.paths if not p.is_file()]
        if missing:
            model = self.paths[0]
            self.problem = (f"{', '.join(missing)} not found in {model.parent} - "
                            + ("train it: scripts/train_wake_model.py" if model.name in missing
                               else "run: scripts/get_wake_model.py"))
            return False
        self.problem = ""
        return True

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise WakeUnavailable(self.problem)
        try:
            import numpy as np

            from nexus.ai.voice.oww import WakeFeatures, WakeModel

            model, melspec, embedding = self.paths
            self._np = np
            self._features = WakeFeatures(melspec, embedding, threads=self.threads)
            self._model = WakeModel(model, threads=self.threads)
        except Exception as exc:  # a broken or incompatible model file
            self._features = self._model = None
            self.problem = f"couldn't load {self.paths[0].name}: {exc}"
            raise WakeUnavailable(self.problem) from None

    def unload(self) -> None:
        self._features = self._model = None

    def start(self) -> None:
        if self.resources is None:
            self.load()
            return
        if not self._registered:
            try:
                self.resources.register(self)  # type: ignore[arg-type]
            except ValueError:
                pass  # already registered
            self._registered = True
        if not self.available():
            raise WakeUnavailable(self.problem)
        hold = ExitStack()
        try:
            hold.enter_context(self.resources.use(self.spec.name))
        except (ResourceUnavailable, WakeUnavailable) as exc:
            hold.close()
            raise WakeUnavailable(str(exc)) from None
        self._hold = hold

    def stop(self) -> None:
        if self._hold is not None:
            self._hold.close()
            self._hold = None

    # --- detection -----------------------------------------------------------------------------------

    def reset(self) -> None:
        if self._features is not None:
            self._features.reset()
        self._chunks = 0
        self._above = 0
        self._recent: deque[bytes] = deque(maxlen=self._lead_chunks)
        self._silence_ms = 0
        self._vad = getattr(self, "_vad", None) or EnergyVAD(self.cfg)
        self.peak_score = 0.0

    @property
    def noise_floor(self) -> float | None:
        return self._vad.noise

    def process(self, frame: bytes) -> WakeHit | None:
        if self._model is None:
            self.load()
        np = self._np
        chunk = np.frombuffer(frame, dtype=np.int16)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) if len(chunk) else 0.0
        self._silence_ms = 0 if self._vad.voiced(rms, speaking=False) else self._silence_ms + self.frame_ms
        self._recent.append(frame)
        self._features.push(chunk)
        self._chunks += 1
        score = self._model.score(self._features.window(self._model.frames))
        if self._chunks <= 5:  # openWakeWord ignores the first frames after a reset
            score = 0.0
        self.last_score = score
        self.peak_score = max(self.peak_score, score)
        self._above = self._above + 1 if score >= self.threshold else 0
        if self._above < self.patience:
            return None
        hit = WakeHit(self.info.name, score, "", b"".join(self._recent), self._silence_ms,
                      noise_floor=self._vad.noise)
        self.reset()
        return hit
