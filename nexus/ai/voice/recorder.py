"""Recording one utterance: wait for speech, stop at the pause after it.

Push-to-talk: recording starts when you ask (/listen, or Enter on an empty
line) and always ends by itself - it can never record forever:

  no speech within no_speech_timeout_s        -> NoSpeech
  speech, then end_silence_ms of quiet         -> the utterance
  max_utterance_s of speech                    -> the utterance, cut off (noted)
  cancel event / Ctrl+C                        -> ListeningCancelled
  no audio at all from the device for a while  -> MicrophoneLost

Speech detection is energy based with an adaptive noise floor: no model and
almost no CPU. Whisper's own voice-activity filter then drops any silence
left in the clip. Audio stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from nexus.ai.voice.base import AudioClip, ListeningCancelled, NoSpeech
from nexus.system.audio.capture import CaptureStream
from nexus.system.audio.errors import MicrophoneLost


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 30
    start_ms: int = 120            # voiced audio within start_window_ms that counts as "speech started"
    start_window_ms: int = 300
    end_silence_ms: int = 900      # quiet after speech that ends the utterance
    pre_roll_ms: int = 450         # audio kept from just before speech was detected
    keep_tail_ms: int = 300        # quiet kept after the last word
    min_speech_ms: int = 200       # shorter bursts (a click, a cough) are ignored
    calibration_ms: int = 300      # first moments used to measure the room's noise
    no_speech_timeout_s: float = 6.0
    max_utterance_s: float = 15.0
    min_level: float = 300.0       # RMS on the 16-bit scale (about -40 dBFS): quieter is never speech
    max_start_noise: float = 600.0  # calibration never assumes the room is louder than about -35 dBFS
    snr: float = 3.0               # speech must be this many times louder than the noise floor
    device_timeout_s: float = 2.0  # no audio at all from the microphone for this long = it is gone


def level(frame: bytes) -> float:
    """RMS of mono PCM16 (0 = digital silence, ~32768 = full scale)."""
    samples = memoryview(frame).cast("h")
    if not len(samples):
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    def __init__(self, cfg: VADConfig) -> None:
        self.cfg = cfg
        self.noise: float | None = None
        self._calibration: list[float] = []

    @property
    def calibrating(self) -> bool:
        return self.noise is None

    def threshold(self, *, speaking: bool = False) -> float:
        base = max(self.cfg.min_level, (self.noise or 0.0) * self.cfg.snr)
        return base * (0.7 if speaking else 1.0)  # a little easier to stay "in speech" than to start it

    def voiced(self, rms: float, *, speaking: bool) -> bool:
        if self.noise is None:
            self._calibration.append(rms)
            if len(self._calibration) * self.cfg.frame_ms >= self.cfg.calibration_ms:
                # The quietest moment - capped, so talking straight away isn't taken for room noise.
                # A really noisy room raises it afterwards (below).
                self.noise = min(min(self._calibration), self.cfg.max_start_noise)
            return False
        is_voiced = rms > self.threshold(speaking=speaking)
        if not is_voiced and not speaking:
            # Follow the room's quietest level: down at once, up slowly (about 2x per second), so a fan
            # that starts is learned but the dips between syllables never make speech look like noise.
            self.noise = rms if rms < self.noise else min(rms, self.noise * 1.02)
        return is_voiced


@dataclass(frozen=True)
class Utterance:
    clip: AudioClip
    speech_seconds: float
    truncated: bool = False    # max_utterance_s reached while still speaking


class Recorder:
    def __init__(self, cfg: VADConfig | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg or VADConfig()
        self._clock = clock

    def record(self, stream: CaptureStream, *, cancel: threading.Event | None = None,
               on_speech: Callable[[], None] | None = None) -> Utterance:
        cfg = self.cfg
        rate = stream.sample_rate
        frame_bytes = max(2, rate * cfg.frame_ms // 1000 * 2)
        vad = EnergyVAD(cfg)
        # Speech can't be detected while the noise is measured, so the lead-in also covers that time.
        pre_roll: deque[bytes] = deque(maxlen=max(1, (cfg.pre_roll_ms + cfg.calibration_ms) // cfg.frame_ms))
        recent: deque[bool] = deque(maxlen=max(1, cfg.start_window_ms // cfg.frame_ms))
        speech: list[bytes] = []
        speaking = False
        voiced_ms = silence_ms = frames = zero_frames = 0
        started_at = 0
        pending = b""
        last_audio = self._clock()
        while True:
            if cancel is not None and cancel.is_set():
                raise ListeningCancelled("Cancelled.")
            block = stream.read(0.1)
            if block is None:
                if self._clock() - last_audio > cfg.device_timeout_s or not stream.active:
                    lost = MicrophoneLost("The microphone stopped sending audio (disconnected?).")
                    lost.partial = speaking  # type: ignore[attr-defined]
                    raise lost
                continue
            last_audio = self._clock()
            pending += block
            while len(pending) >= frame_bytes:
                frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                frames += 1
                rms = level(frame)
                zero_frames += rms == 0.0
                if not speaking:
                    pre_roll.append(frame)
                    recent.append(vad.voiced(rms, speaking=False))
                    if sum(recent) * cfg.frame_ms >= cfg.start_ms:
                        speaking, started_at = True, frames
                        speech = list(pre_roll)
                        voiced_ms, silence_ms = sum(recent) * cfg.frame_ms, 0
                        if on_speech is not None:
                            on_speech()
                    elif frames * cfg.frame_ms >= cfg.no_speech_timeout_s * 1000:
                        raise NoSpeech(self._no_speech_message(zero_frames == frames))
                    continue
                speech.append(frame)
                if vad.voiced(rms, speaking=True):
                    voiced_ms += cfg.frame_ms
                    silence_ms = 0
                else:
                    silence_ms += cfg.frame_ms
                if silence_ms >= cfg.end_silence_ms:
                    if voiced_ms < cfg.min_speech_ms:  # just a click or a cough: keep waiting
                        speaking = False
                        recent.clear()
                        pre_roll.clear()
                        continue
                    drop = (silence_ms - cfg.keep_tail_ms) // cfg.frame_ms
                    kept = speech[:len(speech) - drop] if drop > 0 else speech
                    return Utterance(AudioClip(b"".join(kept), rate), voiced_ms / 1000)
                if (frames - started_at) * cfg.frame_ms >= cfg.max_utterance_s * 1000:
                    return Utterance(AudioClip(b"".join(speech), rate), voiced_ms / 1000, truncated=True)

    def _no_speech_message(self, all_zero: bool) -> str:
        if all_zero:
            return ("The microphone sent only silence - it may be muted, or its volume is at zero in the "
                    "system sound settings.")
        return f"I didn't hear anything in {self.cfg.no_speech_timeout_s:g} seconds."
