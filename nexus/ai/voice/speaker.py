"""Speaker: speaks NEXUS's replies through a TTS provider, resource-aware.

The voice model is loaded only when there is something to say (SPEAKING
state), released afterwards, and unloaded by the ResourceManager when idle.
If the model can't be loaded (e.g. game mode and a GPU voice), NEXUS stays
text-only rather than taking resources from the game.

Planned voice pipeline (subsystems 5-7), each stage behind an interface:

    wake word (tiny, CPU, always on)
      -> LISTENING: STT model loaded on demand
      -> THINKING / EXECUTING: fast path or planner, Action Kernel
      -> SPEAKING: TTS model loaded on demand (this class)
      -> IDLE: STT/TTS unloaded after their idle timeouts
"""

from __future__ import annotations

import logging

from nexus.ai.conversation.types import Say, TurnEvent
from nexus.ai.voice.base import TTSProvider
from nexus.core.resources import ResourceManager, ResourceState, ResourceUnavailable

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, tts: TTSProvider, resources: ResourceManager | None = None) -> None:
        self.tts = tts
        self.resources = resources
        # A TTS with a model declares a ModelSpec (it is a ModelBackend); SilentTTS has none.
        self._managed = resources is not None and hasattr(tts, "spec")
        if self._managed:
            resources.register(tts)  # type: ignore[arg-type]

    def __call__(self, event: TurnEvent) -> None:
        if isinstance(event, Say) and event.text:
            self.say(event.text, language=event.language)

    def say(self, text: str, *, language: str | None = None) -> bool:
        """Speak; returns False if the voice couldn't run right now (the text is still shown)."""
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

    def stop(self) -> None:
        self.tts.stop()
