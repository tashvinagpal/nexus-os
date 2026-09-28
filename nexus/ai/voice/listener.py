"""Voice input: microphone -> one utterance -> text for the normal conversation engine.

A spoken request becomes an ordinary UserInput(interface="voice") and goes
through the same fast path, planner, Action Kernel, policy, confirmations and
audit log as typed text. Voice adds no command system and grants nothing by
itself (a spoken "yes" only counts when it was heard clearly - see
ConversationEngine).

    listen()
      microphone privacy allowed? -> pick a microphone (MicrophoneManager)
      -> pick an STT model and start loading it (HybridSTT.prepare)
      -> LISTENING: record until the pause after speech (Recorder)
           a microphone that won't open or disconnects -> the next one
      -> transcribe (LISTENING), clean, reject noise/unclear speech
      -> normalize: romanize, regional forms, sound-alike words (nexus/ai/voice/normalize.py)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from nexus.ai.conversation.responder import parse_confirmation
from nexus.ai.voice.base import ListeningCancelled, NoSpeech, STTUnavailable, Transcript, Unintelligible
from nexus.ai.voice.normalize import Normalizer
from nexus.ai.voice.recorder import Recorder, Utterance, VADConfig
from nexus.ai.voice.stt import HybridSTT
from nexus.ai.voice.text import clean, likely_invented
from nexus.core.resources import ResourceManager, ResourceState
from nexus.system.audio.capture import CaptureBackend, CaptureStream
from nexus.system.audio.devices import MicrophoneManager, MicScan
from nexus.system.audio.errors import (MicrophoneDenied, MicrophoneLost, MicrophoneUnavailable, NoMicrophone)
from nexus.system.platform.base import AudioInput

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Heard:
    text: str                 # what NEXUS acts on (romanized/normalized - see normalize.py)
    original: str             # what the model wrote
    transcript: Transcript
    microphone: str
    speech_seconds: float
    truncated: bool = False
    guessed: bool = False     # some words were replaced by similar-sounding ones: NEXUS asks before acting
    changes: tuple[str, ...] = ()

    @property
    def confidence(self) -> float | None:
        return self.transcript.confidence


class VoiceInput:
    def __init__(self, mics: MicrophoneManager, capture: CaptureBackend, stt: HybridSTT,
                 resources: ResourceManager, *, vad: VADConfig | None = None,
                 permission: Callable[[], bool | None] | None = None, min_confidence: float = 0.4,
                 romanize: bool = True, language: str | None = None, recorder: Recorder | None = None,
                 device_poll_s: float = 5.0, fast_path: Callable[[str], bool] | None = None,
                 names: Callable[[], Iterable[str]] | None = None) -> None:
        self.mics = mics
        self.capture = capture
        self.stt = stt
        self.resources = resources
        self.vad = vad or VADConfig()
        self.recorder = recorder or Recorder(self.vad)
        self._permission = permission
        self.min_confidence = min_confidence
        self.romanize = romanize
        self.language = language
        self.device_poll_s = device_poll_s  # how often the front end has the device list rechecked
        # fast_path: does the fast path understand this text? names: installed apps' names (sound-alikes).
        # A yes/no answer is understood too (the conversation engine decides what a yes may approve).
        self.normalizer = Normalizer(
            lambda text: bool(fast_path and fast_path(text)) or parse_confirmation(text) is not None, names)
        self.last_timings: dict[str, float] = {}
        self.last_utterance: Utterance | None = None

    def listen(self, *, cancel: threading.Event | None = None,
               on_status: Callable[[str], None] = lambda _: None) -> Heard:
        if self._permission is not None and self._permission() is False:
            raise MicrophoneDenied("Microphone access is turned off. Allow it in Windows Settings > Privacy & "
                                   "security > Microphone (\"Let desktop apps access your microphone\").")
        scan = self.mics.scan()
        if not scan.candidates:
            raise NoMicrophone(self._no_microphone(scan))
        if self.stt.prepare() is None:  # say so before you speak, not after
            _, reasons = self.stt.choose()
            raise STTUnavailable("I can't transcribe speech right now: " + "; ".join(reasons))
        started = time.perf_counter()
        try:
            with self.resources.busy(ResourceState.LISTENING):
                utterance, mic = self._record(scan, cancel, on_status)
                recorded = time.perf_counter()
                on_status("transcribing")
                transcript = self.stt.transcribe(utterance.clip, language_hint=self.language, cancel=cancel)
        except KeyboardInterrupt:
            raise ListeningCancelled("Cancelled.") from None
        done = time.perf_counter()
        self.last_timings = {"recording_s": recorded - started, "transcribing_s": done - recorded,
                             "audio_s": utterance.clip.seconds}
        return self._heard(utterance, transcript, mic)

    def listen_on(self, stream: CaptureStream, mic: AudioInput, *, cancel: threading.Event | None = None,
                  no_speech_timeout_s: float | None = None, **record: Any) -> Heard:
        """One utterance from a stream that is already open (after the wake word - see wake_listener.py).
        `record` goes to Recorder.record (noise_floor, lead_in, lead_in_silence_ms, pre_roll)."""
        recorder = self.recorder
        if no_speech_timeout_s is not None:
            recorder = Recorder(replace(self.vad, no_speech_timeout_s=no_speech_timeout_s))
        started = time.perf_counter()
        try:
            with self.resources.busy(ResourceState.LISTENING):
                self.stt.prepare()
                utterance = recorder.record(stream, cancel=cancel, **record)
                recorded = time.perf_counter()
                transcript = self.stt.transcribe(utterance.clip, language_hint=self.language, cancel=cancel)
        except KeyboardInterrupt:
            raise ListeningCancelled("Cancelled.") from None
        self.mics.mark_working(mic)
        self.last_timings = {"recording_s": recorded - started, "transcribing_s": time.perf_counter() - recorded,
                             "audio_s": utterance.clip.seconds}
        return self._heard(utterance, transcript, mic)

    def _heard(self, utterance: Utterance, transcript: Transcript, mic: AudioInput) -> Heard:
        self.last_utterance = utterance  # in memory only, for diagnostics (e.g. speech level)
        original = self._accept(transcript)
        understood = self.normalizer(original) if self.romanize else None
        return Heard(understood.text if understood else original, original, transcript, mic.name,
                     utterance.speech_seconds, utterance.truncated,
                     guessed=bool(understood and understood.guessed),
                     changes=understood.changes if understood else ())

    # --- microphones -----------------------------------------------------------------

    def _record(self, scan: MicScan, cancel: threading.Event | None,
                on_status: Callable[[str], None]) -> tuple[Utterance, AudioInput]:
        candidates = list(scan.candidates)
        problems: list[str] = []
        while candidates:
            mic = candidates.pop(0)
            try:
                stream = self.capture.open(mic, sample_rate=16000, block_ms=self.vad.frame_ms)
            except MicrophoneUnavailable as exc:
                self.mics.mark_failed(mic, str(exc))
                problems.append(str(exc))
                if candidates:
                    on_status(f"couldn't open {mic.name} - trying {candidates[0].name}")
                continue
            on_status(f"listening with {mic.name}")
            try:
                utterance = self.recorder.record(stream, cancel=cancel)
                self.mics.mark_working(mic)
                return utterance, mic
            except MicrophoneLost as exc:
                self.mics.mark_failed(mic, "stopped sending audio")
                candidates = [c for c in self.mics.scan().candidates if c.id != mic.id]
                if getattr(exc, "partial", False):  # you were mid-sentence: don't guess the rest
                    raise MicrophoneLost(
                        f"{mic.name} stopped while you were speaking. "
                        + (f"Switched to {candidates[0].name} - please say it again."
                           if candidates else "No other microphone is available.")) from None
                if candidates:
                    on_status(f"{mic.name} disconnected - listening with {candidates[0].name}")
                problems.append(f"{mic.name} stopped sending audio")
            finally:
                stream.close()
        if problems and not any("stopped" in p for p in problems):
            raise MicrophoneUnavailable("No microphone could be opened: " + "; ".join(problems))
        raise NoMicrophone("The microphone was disconnected and no other one is available.")

    def _no_microphone(self, scan: MicScan) -> str:
        if scan.error:
            return f"I couldn't check for microphones: {scan.error}."
        message = "No microphone is connected, so voice input is off for now (typing still works)."
        virtual = [d.name for d, why in scan.ignored if "virtual" in why]
        if virtual:
            message += f" Virtual inputs aren't used automatically ({', '.join(virtual)}); /mic use <name> picks one."
        return message + " Plug one in and NEXUS will find it - no restart needed."

    # --- transcripts ------------------------------------------------------------------

    def _accept(self, transcript: Transcript) -> str:
        """The transcript as written, or an error if it's nothing, noise, or too unclear to act on."""
        text = clean(transcript.text)
        confidence = transcript.confidence
        no_speech = transcript.no_speech_probability or 0.0
        if not text:
            raise NoSpeech("I didn't catch any words.")
        if likely_invented(text) and (confidence is None or confidence < 0.7 or no_speech > 0.3):
            raise NoSpeech("I only heard noise.")
        if confidence is not None and confidence < self.min_confidence:
            raise Unintelligible(f"I couldn't make that out clearly (I heard \"{text}\"). Please say it again.")
        return text
