"""Speaker - the TTS manager: speaks NEXUS's replies, resource-aware and interruptible.

    reply (a Say event, already shown on screen)
      -> prepare the text for speech (speech_text.py): no markdown, links, code...
      -> sentence chunks
      -> SPEAKING: voice model loaded on demand (the ResourceManager admits it)
           synthesize sentence 1 -> play it; while it plays, synthesize sentence 2 ...
      -> IDLE: the model unloads after its idle timeout

It is an output layer only: it receives text that was already decided and
shown, never produces requests, and has no way to run anything. Every failure
(no model, no audio device, a crashed engine) is logged and the text stays on
screen; nothing here can stop the assistant from working.

- stop() silences NEXUS at once and drops anything queued. Front ends call it
  whenever the user starts something new, so talking or typing interrupts it.
- It never talks over the user: while a microphone is recording (LISTENING) it waits.
- In game mode it stays quiet by default ([tts] game_mode) and frees its model.
- Providers that only know speak() (the original interface, e.g. SilentTTS)
  still work: they are called directly, synchronously.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

from nexus.ai.conversation.types import Say, TurnEvent
from nexus.ai.voice.base import AudioClip, TTSError, TTSProvider
from nexus.ai.voice.speech_text import prepare_for_speech
from nexus.core.resources import ResourceManager, ResourceState, ResourceUnavailable

log = logging.getLogger(__name__)

_DEVANAGARI = ("ऀ", "ॿ")
_GURMUKHI = ("਀", "੿")


@dataclass(frozen=True)
class SpeechSettings:
    enabled: bool = True
    max_chars: int = 600          # longer replies: the first sentences, then "The rest is on the screen."
    chunk_chars: int = 220
    game_mode: str = "quiet"      # quiet: text only while a game runs | speak: keep talking
    romanized: str = "hindi"      # Hinglish/Punjabi in Latin letters: hindi (converted for the Hindi voice) | english
    listening_wait_s: float = 30.0


def _has(text: str, block: tuple[str, str]) -> bool:
    return any(block[0] <= ch <= block[1] for ch in text)


class Speaker:
    def __init__(self, tts: TTSProvider, resources: ResourceManager | None = None, *, sink: Any = None,
                 settings: SpeechSettings | None = None, clock: Callable[[], float] = time.monotonic,
                 names: Callable[[], list[str]] | None = None) -> None:
        self.tts = tts
        self.resources = resources
        self.sink = sink
        self.settings = settings or SpeechSettings()
        self.enabled = self.settings.enabled
        self._clock = clock
        self._names = names  # installed apps' names: they stay English inside Hinglish speech
        # A TTS with a model declares a ModelSpec (it is a ModelBackend); SilentTTS has none.
        self._managed = resources is not None and hasattr(tts, "spec")
        if self._managed:
            resources.register(tts)  # type: ignore[arg-type]
        self._synthesizes = hasattr(tts, "synthesize") and sink is not None
        self._generation = 0
        self._queue: queue.Queue[tuple[int, str, str | None]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._busy = threading.Event()  # an utterance is being prepared, synthesized or played
        self.state = "idle"             # idle | waiting | synthesizing | playing
        self.problem = ""
        self.last: dict[str, Any] = {}
        if self._synthesizes:
            log.info("voice output: %s", tts.info.name)

    # --- front-end API ------------------------------------------------------------------

    def __call__(self, event: TurnEvent) -> None:
        if isinstance(event, Say) and event.text:
            self.say_async(event.text, language=event.language)

    @property
    def speaking(self) -> bool:
        return self._busy.is_set() or not self._queue.empty()

    def say(self, text: str, *, language: str | None = None) -> bool:
        """Speak and return when done. False if nothing was spoken (the text is still shown)."""
        if not self.enabled or not text:
            return False
        if not self._synthesizes:
            return self._say_legacy(text, language)
        return self._speak(text, language, self._generation)

    def say_async(self, text: str, *, language: str | None = None) -> None:
        """Queue text to be spoken in the background; returns at once."""
        if not self.enabled or not text:
            return
        if not self._synthesizes:  # SilentTTS and speak()-only providers: nothing to wait for
            try:
                self._say_legacy(text, language)
            except Exception:
                log.exception("speaking failed")
            return
        self._queue.put((self._generation, text, language))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="nexus-speaker", daemon=True)
            self._thread.start()

    def stop(self) -> bool:
        """Silence NEXUS now and forget anything queued. True if it was speaking."""
        was = self.speaking
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        started = self._clock()
        if self.sink is not None:
            try:
                self.sink.stop()
            except Exception:
                log.exception("stopping playback failed")
        try:
            self.tts.stop()
        except Exception:
            log.exception("stopping the voice failed")
        if was:
            self.last["stop_s"] = round(self._clock() - started, 4)
            log.info("speech stopped")
        return was

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until everything queued has been spoken. False on timeout."""
        deadline = None if timeout is None else self._clock() + timeout
        while self.speaking:
            if deadline is not None and self._clock() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def prepare(self) -> None:
        """Start loading the voice in the background (e.g. while NEXUS is thinking)."""
        if not (self._synthesizes and self._managed and self.enabled) or self.tts.is_loaded():  # type: ignore[attr-defined]
            return
        if self.resources is not None and self.resources.game_mode and self.settings.game_mode == "quiet":
            return

        def load() -> None:
            try:
                with self.resources.use(self.tts.spec.name):  # type: ignore[union-attr, attr-defined]
                    pass
            except Exception as exc:
                log.info("preloading the voice failed: %s", exc)

        threading.Thread(target=load, name="nexus-voice-preload", daemon=True).start()

    def status(self) -> dict[str, Any]:
        ready, device = (self.sink.ready() if self.sink is not None else (False, "no audio output configured"))
        return {"enabled": self.enabled, "provider": self.tts.info.name, "state": self.state,
                "loaded": bool(getattr(self.tts, "is_loaded", lambda: False)()),
                "output": device if ready else None, "output_problem": None if ready else device,
                "problem": self.problem, "last": dict(self.last)}

    # --- internals ------------------------------------------------------------------------

    def _say_legacy(self, text: str, language: str | None) -> bool:
        if not self._managed:
            self.tts.speak(text, language=language)
            return True
        assert self.resources is not None
        try:
            with self.resources.busy(ResourceState.SPEAKING), self.resources.use(self.tts.spec.name):  # type: ignore[attr-defined]
                self.tts.speak(text, language=language)
            return True
        except ResourceUnavailable as exc:
            log.info("speaking skipped: %s", exc)
            return False

    def _run(self) -> None:
        while True:
            try:
                generation, text, language = self._queue.get(timeout=5)
            except queue.Empty:
                return  # the thread ends when idle; say_async starts a new one
            if generation != self._generation:
                continue  # stopped before its turn
            try:
                self._speak(text, language, generation)
            except Exception:  # never let the speaker thread die
                log.exception("speaking failed")

    def _voice_for(self, text: str, language: str | None) -> tuple[str, str, bool]:
        """(provider language, text language, convert romanized Hindi to Devanagari) for this reply.

        Kokoro has English and Hindi voices. Hindi in Devanagari -> the Hindi voice. Punjabi in
        Gurmukhi -> converted to Devanagari for the Hindi voice (an approximation: there is no
        Punjabi voice). Hinglish/Punjabi in Latin letters -> converted too, because an English
        voice can't pronounce it ([tts] romanized = "english" reads it with the English voice).
        """
        if _has(text, _GURMUKHI):
            return "hi", "pa", False
        if _has(text, _DEVANAGARI):
            return "hi", "hi", False
        if language in ("hinglish", "pa"):
            return ("hi", language, True) if self.settings.romanized == "hindi" else ("en", language, False)
        return "en", "en", False

    def _problem(self, message: str) -> None:
        if message != self.problem:
            log.warning("voice output: %s", message)
        self.problem = message

    def _speak(self, text: str, language: str | None, generation: int) -> bool:
        self._busy.set()
        try:
            return self._speak_now(text, language, generation)
        finally:
            if self._queue.empty():
                self._busy.clear()

    def _speak_now(self, text: str, language: str | None, generation: int) -> bool:
        started = self._clock()
        tts_language, text_language, convert = self._voice_for(text, language)
        voice = None  # the provider's voice for that language
        if _has(text, _GURMUKHI):
            from nexus.ai.voice.text import _to_devanagari

            text = _to_devanagari(text)
        speech = prepare_for_speech(text, language=text_language, max_chars=self.settings.max_chars,
                                    chunk_chars=self.settings.chunk_chars)
        if not speech.chunks:
            return False
        chunks = speech.chunks
        if convert:
            from nexus.ai.voice.hinglish import to_devanagari_speech

            names = self._names() if self._names is not None else ()
            chunks = tuple(to_devanagari_speech(chunk, names=names) for chunk in chunks)
        if self.resources is not None and self.resources.game_mode and self.settings.game_mode == "quiet":
            if self._managed and self.resources.unload(self.tts.spec.name):  # type: ignore[attr-defined]
                log.info("voice model unloaded for game mode")
            return False
        ready, why = self.sink.ready()
        if not ready:
            self._problem(why)
            return False
        # Never talk over the user while they are speaking into the microphone.
        self.state = "waiting"
        waited = self._clock()
        while (self.resources is not None and self.resources.activity is ResourceState.LISTENING
               and generation == self._generation and self._clock() - waited < self.settings.listening_wait_s):
            time.sleep(0.05)
        if generation != self._generation:
            self.state = "idle"
            return False
        timings: dict[str, Any] = {"characters": sum(len(c) for c in chunks), "chunks": len(chunks),
                                   "truncated": speech.truncated, "language": tts_language, "stopped": False}
        try:
            with ExitStack() as stack:
                if self.resources is not None:
                    stack.enter_context(self.resources.busy(ResourceState.SPEAKING))
                if self._managed:
                    was_loaded = self.tts.is_loaded()  # type: ignore[attr-defined]
                    load_started = self._clock()
                    stack.enter_context(self.resources.use(self.tts.spec.name))  # type: ignore[union-attr, attr-defined]
                    if not was_loaded:
                        timings["load_s"] = round(self._clock() - load_started, 3)
                self.state = "synthesizing"
                t = self._clock()
                clip = self._synthesize(chunks[0], tts_language, voice)
                timings["first_synthesis_s"] = round(self._clock() - t, 3)
                for index in range(len(chunks)):
                    if generation != self._generation:
                        timings["stopped"] = True
                        return False
                    self.state = "playing"
                    playback = self.sink.play(clip)
                    next_clip: AudioClip | None = None
                    if index + 1 < len(chunks):  # synthesize the next sentence while this one plays
                        next_clip = self._synthesize(chunks[index + 1], tts_language, voice)
                    while not playback.wait(0.05):
                        if generation != self._generation:
                            playback.stop()
                            break
                    if index == 0 and playback.started_at is not None:
                        timings["first_audio_s"] = round(playback.started_at - started, 3)
                    if playback.error:
                        self._problem(playback.error)
                        return False
                    if generation != self._generation:
                        timings["stopped"] = True
                        return False
                    if next_clip is not None:
                        clip = next_clip
            self.problem = ""
            return True
        except (ResourceUnavailable, TTSError) as exc:
            self._problem(str(exc))
            return False
        except Exception as exc:  # an audio device error, a broken driver...
            from nexus.system.audio.playback import AudioOutputError

            if isinstance(exc, AudioOutputError):
                self._problem(str(exc))
                return False
            raise
        finally:
            timings["total_s"] = round(self._clock() - started, 3)
            self.last = timings
            self.state = "idle"
            log.info("spoke %s characters in %s chunk(s) (%s)", timings["characters"], timings["chunks"],
                     "stopped" if timings["stopped"] else f"{timings['total_s']} s")

    def _synthesize(self, chunk: str, language: str, voice: str | None) -> AudioClip:
        return self.tts.synthesize(chunk, language=language, voice=voice)  # type: ignore[attr-defined]
