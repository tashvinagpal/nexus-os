"""The always-on part of voice input: wait for "Nexus", then hear the request.

    WakeListener thread
      pick a microphone (MicrophoneManager, same rules as push-to-talk) and open it
      -> every frame to the WakeDetector (wake.py); nothing is stored or sent anywhere
      -> hit: stop NEXUS talking, keep recording the same utterance (Recorder)
           "Nexus, GTA chala de"  -> "GTA chala de" -> on_heard (the conversation engine)
           "Nexus" ... pause       -> a short chime, then the request is recorded
      -> NEXUS asked something (yes/no, which one?) -> the answer is heard without "Nexus"
      -> back to waiting

It pauses while push-to-talk records, while NEXUS itself speaks (so it doesn't
wake itself up, unless barge_in is on), and in game mode if configured. A
microphone that disappears is replaced by the next one; one plugged in later is
picked up. Every failure is reported once and retried - it never ends NEXUS.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Sequence

from nexus.ai.voice.base import AudioClip, ListeningCancelled, NoSpeech, VoiceInputError
from nexus.ai.voice.wake import RATE, WakeDetector, WakeHit, WakeUnavailable, find_wake
from nexus.core.resources import ResourceManager
from nexus.system.audio.errors import AudioInputError, MicrophoneLost, MicrophoneUnavailable
from nexus.system.platform.base import AudioInput

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeOptions:
    command_timeout_s: float = 6.0   # after "Nexus" on its own: how long to wait for the request
    follow_up_s: float = 6.0         # after NEXUS asks something: answer without "Nexus" (0 = off)
    max_follow_ups: int = 3
    barge_in: bool = False           # listen for "Nexus" while NEXUS is speaking
    game_mode: str = "listen"        # listen | off (pause while a game runs)
    chime: bool = True
    speaking_tail_s: float = 0.4     # NEXUS's own echo after it stops talking
    retry_s: float = 5.0             # no microphone / detector: look again after this long
    device_timeout_s: float = 2.0


@dataclass(frozen=True)
class WakeEvent:
    kind: str       # listening | woke | heard | nothing | problem | test | follow_up | stopped
    message: str = ""
    hit: WakeHit | None = None


def chime_clip(volume: float = 0.18) -> AudioClip:
    """Two short rising tones (~0.2 s): "I'm listening"."""
    samples = array("h")
    for freq, ms in ((660, 70), (990, 110)):
        n = RATE * ms // 1000
        for i in range(n):
            fade = min(1.0, i / 160, (n - i) / 160)
            samples.append(int(32767 * volume * fade * math.sin(2 * math.pi * freq * i / RATE)))
    return AudioClip(samples.tobytes(), RATE)


class WakeListener:
    def __init__(self, voice: Any, detectors: Sequence[WakeDetector], resources: ResourceManager, *,
                 on_heard: Callable[[Any, WakeHit | None], bool], on_event: Callable[[WakeEvent], None] = lambda e: None,
                 speaking: Callable[[], bool] = lambda: False, stop_speaking: Callable[[], Any] = lambda: None,
                 chime: Callable[[], None] | None = None, options: WakeOptions | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.voice = voice
        self.detectors = list(detectors)  # preference order; the first that starts is used
        self.resources = resources
        self.on_heard = on_heard
        self.on_event = on_event
        self.speaking = speaking
        self.stop_speaking = stop_speaking
        self.chime = chime
        self.options = options or WakeOptions()
        self._clock = clock
        self.enabled = True
        self.detector: WakeDetector | None = None
        self.mic: AudioInput | None = None
        self.problem = ""
        self.wakes = 0
        self.skipped: list[str] = []   # detectors that couldn't start, and why (e.g. no trained model yet)
        self._stream: Any = None
        self._pauses: dict[object, str] = {}
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._thread: threading.Thread | None = None
        self._test_until = 0.0
        self._opened_at = 0.0
        self._announced: tuple[str, str] | None = None
        self._reported = ""

    # --- control (any thread) ---------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nexus-wake", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._poke.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._reported = ""
        self._announced = None
        self._poke.set()

    @contextmanager
    def paused(self, reason: str) -> Iterator[None]:
        """No wake word while this block runs (e.g. push-to-talk has the microphone)."""
        token = object()
        self._pauses[token] = reason
        try:
            yield
        finally:
            self._pauses.pop(token, None)
            self._poke.set()

    def test(self, seconds: float = 20.0) -> None:
        """For the next few seconds, report every detection instead of acting on it."""
        self._test_until = self._clock() + seconds
        self._poke.set()

    @property
    def testing(self) -> bool:
        return self._clock() < self._test_until

    @property
    def pause_reason(self) -> str:
        if not self.enabled:
            return "turned off (/wake on)"
        if self._pauses:
            return next(iter(self._pauses.values()))
        if self.resources.game_mode and self.options.game_mode == "off":
            return "a game is running"
        return ""

    def status(self) -> dict[str, Any]:
        state = "off" if not self.enabled else ("paused" if self.pause_reason else
                                                ("listening" if self._stream is not None else "starting"))
        if self.enabled and self.problem and self._stream is None:
            state = "unavailable"
        return {"state": state, "detector": self.detector.info.name if self.detector else None,
                "microphone": self.mic.name if self.mic else None, "problem": self.problem,
                "paused": self.pause_reason, "wakes": self.wakes, "testing": self.testing}

    # --- the thread ---------------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._step()
                except Exception as exc:  # never let the wake word take NEXUS down
                    log.exception("wake word listener failed")
                    self._close()
                    self._report(f"the wake word stopped unexpectedly ({type(exc).__name__}: {exc}); retrying")
                    self._wait(self.options.retry_s)
        finally:
            self._close()
            if self.detector is not None:
                self.detector.stop()

    def _wait(self, seconds: float) -> None:
        self._poke.wait(seconds)
        self._poke.clear()

    def _report(self, problem: str) -> None:
        self.problem = problem
        if problem and problem != self._reported:
            self._reported = problem
            self.on_event(WakeEvent("problem", problem))

    def _step(self) -> None:
        if self.pause_reason:
            self._close()
            self._wait(0.5)
            return
        if self.detector is None and not self._start_detector():
            self._wait(self.options.retry_s * 6)
            return
        if self._stream is None and not self._open():
            self._wait(self.options.retry_s)
            return
        self._listen()

    def _start_detector(self) -> bool:
        problems = []
        for detector in self.detectors:
            try:
                detector.start()
            except WakeUnavailable as exc:
                problems.append(f"{detector.info.name}: {exc}")
                continue
            self.detector = detector
            self.skipped = problems
            log.info("wake word detector: %s", detector.info.name)
            return True
        self._report("no wake word detector can run - " + ("; ".join(problems) or "none is configured"))
        return False

    def _open(self) -> bool:
        permission = getattr(self.voice, "_permission", None)
        if permission is not None and permission() is False:
            self._report("microphone access is turned off in Windows Settings > Privacy & security > Microphone")
            return False
        mics = self.voice.mics
        scan = mics.scan()
        if not scan.candidates:
            self._report("no microphone is connected (it will be used as soon as one is plugged in)")
            return False
        problems = []
        for mic in scan.candidates:
            try:
                self._stream = self.voice.capture.open(mic, sample_rate=RATE, block_ms=30)
            except (MicrophoneUnavailable, AudioInputError) as exc:
                mics.mark_failed(mic, str(exc))
                problems.append(str(exc))
                continue
            self.mic = mic
            self._opened_at = self._clock()
            self.problem = self._reported = ""
            assert self.detector is not None
            self.detector.reset()
            now = (mic.name, self.detector.info.name)
            if now != self._announced:
                self._announced = now
                self.on_event(WakeEvent("listening", f'listening for "Nexus" with {mic.name}'))
            return True
        self._report("no microphone could be opened: " + "; ".join(problems))
        return False

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _listen(self) -> None:
        """Feed audio to the detector until something needs the loop's attention."""
        detector, stream = self.detector, self._stream
        assert detector is not None and stream is not None
        frame_bytes = RATE * detector.frame_ms // 1000 * 2
        pending = b""
        last_audio = self._clock()
        quiet_until = 0.0      # NEXUS was speaking: ignore its echo until then
        was_speaking = False
        next_check = self._clock() + 2.0
        while not self._stop.is_set() and not self.pause_reason and not self._poke.is_set():
            now = self._clock()
            if now >= next_check:
                next_check = now + 2.0
                chosen = (self.voice.mics.last_scan or self.voice.mics.scan()).chosen
                if chosen is not None and self.mic is not None and chosen.id != self.mic.id:
                    log.info("wake word: switching to %s", chosen.name)
                    self._close()  # a better (or the picked) microphone is available
                    return
            block = stream.read(0.1)
            if block is None:
                if not stream.active or self._clock() - last_audio > self.options.device_timeout_s:
                    self._lost()
                    return
                continue
            last_audio = self._clock()
            if self.speaking() and not self.options.barge_in:
                was_speaking = True
                pending = b""
                continue
            if was_speaking:
                was_speaking = False
                quiet_until = self._clock() + self.options.speaking_tail_s
                detector.reset()
            if self._clock() < quiet_until:
                continue
            pending += block
            while len(pending) >= frame_bytes:
                frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                hit = detector.process(frame)
                if hit is not None:
                    self._handle(hit)
                    return
        self._poke.clear()

    def _lost(self) -> None:
        """The stream ended. After a while of working that is usually not the microphone's fault (the audio
        system was re-initialised for a new speaker/headphones): reopen it. Right after opening: skip it."""
        mic = self.mic
        self._close()
        if mic is not None and self._clock() - self._opened_at < 5.0:
            self.voice.mics.mark_failed(mic, "stopped sending audio")
            log.info("wake word: %s stopped sending audio", mic.name)
        elif mic is not None:
            log.info("wake word: %s stream ended - reopening", mic.name)

    # --- after "Nexus" ---------------------------------------------------------------------------------

    def _handle(self, hit: WakeHit) -> None:
        assert self.detector is not None
        detector = self.detector
        if self.testing:
            score = f"{hit.score:.2f}" if hit.score is not None else "?"
            heard = f' - heard "{hit.heard}"' if hit.heard else ""
            self.on_event(WakeEvent("test", f"wake word detected (score {score}){heard} - test mode, not acting",
                                    hit))
            detector.reset()
            return
        self.wakes += 1
        self.stop_speaking()
        self.on_event(WakeEvent("woke", "", hit))
        stream, mic = self._stream, self.mic
        skip = 0 if hit.heard else 2  # the audio kept before an openWakeWord hit may start mid-sentence
        try:
            if hit.alone:
                heard = self._request(stream, mic, hit)
            else:
                heard = self.voice.listen_on(stream, mic, noise_floor=hit.noise_floor, lead_in=hit.lead_in,
                                             lead_in_silence_ms=hit.silence_ms)
                found, rest = find_wake(heard.text, skip=skip)
                if found and not rest:  # only "Nexus" - the request comes after a pause
                    heard = self._request(stream, mic, hit)
                elif found:
                    heard = replace(heard, text=rest, original=find_wake(heard.original, skip=skip)[1] or rest)
            if heard is None:
                return
            follow = self.on_heard(heard, hit)
            for _ in range(self.options.max_follow_ups):
                if not follow or self.options.follow_up_s <= 0 or self._stop.is_set():
                    break
                self._wait_until_quiet(stream)
                self.on_event(WakeEvent("follow_up", "listening for your answer"))
                try:
                    heard = self.voice.listen_on(stream, mic, no_speech_timeout_s=self.options.follow_up_s,
                                                 noise_floor=detector.noise_floor)
                except NoSpeech:
                    break
                follow = self.on_heard(heard, None)
        except ListeningCancelled:
            pass
        except MicrophoneLost as exc:
            self.on_event(WakeEvent("nothing", str(exc)))
            self._lost()
            return
        except (VoiceInputError, AudioInputError) as exc:
            self.on_event(WakeEvent("nothing", str(exc)))
        finally:
            detector.reset()
        self._drain(stream)

    def _request(self, stream: Any, mic: AudioInput, hit: WakeHit):
        """"Nexus" on its own: chime, then record the request."""
        if self.chime is not None and self.options.chime:
            try:
                self.chime()
            except Exception:
                log.exception("wake chime failed")
        self.on_event(WakeEvent("follow_up", "yes? - listening"))
        try:
            return self.voice.listen_on(stream, mic, no_speech_timeout_s=self.options.command_timeout_s,
                                        noise_floor=hit.noise_floor)
        except NoSpeech:
            self.on_event(WakeEvent("nothing", "I didn't hear a request after \"Nexus\"."))
            return None

    def _wait_until_quiet(self, stream: Any, limit_s: float = 60.0) -> None:
        """Let NEXUS finish its question (its own voice is read and thrown away), then a short echo tail."""
        deadline = self._clock() + limit_s
        while self.speaking() and self._clock() < deadline and not self._stop.is_set():
            stream.read(0.05)
        tail = self._clock() + self.options.speaking_tail_s
        while self._clock() < tail:
            stream.read(0.05)
        self._drain(stream)

    @staticmethod
    def _drain(stream: Any) -> None:
        """Drop audio that queued up while NEXUS was thinking or talking."""
        if stream is None:
            return
        for _ in range(1000):
            if stream.read(0) is None:
                return
